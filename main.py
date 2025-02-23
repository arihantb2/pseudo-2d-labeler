import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from colormap import create_semkitti_label_colormap  # provided colormap (&#8203;:contentReference[oaicite:2]{index=2})

#############################################
# PERSPECTIVE PROJECTION & PSEUDO-LABELING   #
#############################################

def project_lidar_to_image(velo, K, T):
    """
    Projects LiDAR points into the camera image plane.
    
    Args:
        velo: LiDAR point cloud as a (N,4) numpy array (first three columns are xyz).
        K: Camera intrinsics matrix (3x3 numpy array).
        T: Camera extrinsics matrix (4x4 numpy array) that transforms LiDAR points to camera coordinates.
        
    Returns:
        proj: The projected points as (N,3), where each row is [u, v, depth].
        valid: Boolean mask of points with positive depth.
    """
    # Convert to homogeneous coordinates
    ones = np.ones((velo.shape[0], 1), dtype=np.float32)
    points_hom = np.hstack([velo[:, :3], ones])  # shape (N,4)
    
    # Transform LiDAR points to camera coordinates
    points_cam = (T @ points_hom.T).T  # shape (N,4)
    
    # Only consider points in front of the camera (positive depth)
    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    
    # Project onto image plane using intrinsics
    proj = (K @ points_cam[:, :3].T).T  # shape (N,3)
    proj = proj / proj[:, 2:3]
    return proj, valid

def generate_pseudo_label(image_path, velo_path, velo_label_path, K_path, T_path, sam_path):
    """
    Generates 2D pseudo-labels for a camera image by projecting LiDAR labels and aggregating them per SAM segment.
    
    Args:
        image_path: Path to the RGB camera image.
        velo_path: Path to the binary LiDAR point cloud file.
        velo_label_path: Path to the binary LiDAR semantic label file.
        K_path: Path to the camera intrinsics (npy file).
        T_path: Path to the camera extrinsics (npy file).
        sam_path: Path to the SAM segments image (PNG file).
        
    Returns:
        pseudo_label: A 2D numpy array (H x W) with the estimated semantic labels.
    """
    # Load image to get dimensions
    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    
    # Load LiDAR points and their semantic labels
    velo = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    velo_label = np.fromfile(velo_label_path, dtype=np.uint32)
    
    # Load camera calibration matrices
    K = np.load(K_path)
    T = np.load(T_path)
    
    # Load SAM segments (each pixel value is the segment id; 0 means ignore)
    sam_segments = cv2.imread(sam_path, cv2.IMREAD_UNCHANGED)
    if sam_segments is None:
        raise ValueError("Error loading SAM segments from " + sam_path)
    
    # Project LiDAR points into image
    proj, valid = project_lidar_to_image(velo, K, T)
    # Only use the labels corresponding to the valid points
    velo_label = velo_label[valid]
    
    # Convert projected coordinates to integer pixel indices
    u = np.round(proj[:, 0]).astype(np.int32)
    v = np.round(proj[:, 1]).astype(np.int32)
    
    # Filter out points that fall outside the image bounds
    valid_idx = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_idx]
    v = v[valid_idx]
    lidar_labels = velo_label[valid_idx]
    
    # For each SAM segment, collect LiDAR labels from projected points
    segment_label_dict = {}
    for i in range(len(u)):
        seg_id = sam_segments[v[i], u[i]]
        if seg_id == 0:
            continue  # ignore pixels not belonging to any segment
        segment_label_dict.setdefault(seg_id, []).append(lidar_labels[i])
    
    # For each segment, assign the most frequent LiDAR label (argmax over the counts)
    segment_pseudo_label = {}
    for seg_id, labels in segment_label_dict.items():
        counts = np.bincount(np.array(labels, dtype=np.int32))
        pseudo = np.argmax(counts)
        segment_pseudo_label[seg_id] = pseudo
        
    # Create the pseudo-label image: assign the computed label to all pixels of each segment
    pseudo_label = np.zeros((H, W), dtype=np.uint8)
    unique_segments = np.unique(sam_segments)
    for seg_id in unique_segments:
        if seg_id == 0:
            continue
        mask = (sam_segments == seg_id)
        # If no LiDAR point falls in the segment, assign label 0 (could be refined)
        pseudo_label[mask] = segment_pseudo_label.get(seg_id, 0)
        
    return pseudo_label

#############################################
# DATASET & MLP MODEL FOR SEGMENTATION       #
#############################################

class SemanticKittiDataset(Dataset):
    """
    A dataset that provides downsampled DINO features and corresponding pseudo-labels.
    The pseudo-labels are downsampled to match the resolution of the DINO features.
    """
    def __init__(self, scan_list, downsample_factor=5.4):
        """
        Args:
            scan_list: List of dictionaries. Each dictionary should have keys:
               'dino': path to the dino_features .npy file.
               'pseudo': path to the corresponding pseudo_label .npy file.
            downsample_factor: Factor by which the dino features are downsampled relative to the original image.
        """
        self.scan_list = scan_list
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, idx):
        scan = self.scan_list[idx]
        # Load DINO features (expected shape: (H_d, W_d, 384))
        dino_feat = np.load(scan['dino'])
        # Load pseudo-label (at original image resolution)
        pseudo_label = np.load(scan['pseudo'])
        
        # Downsample pseudo_label to match dino features using nearest neighbor interpolation
        H_d, W_d, _ = dino_feat.shape
        pseudo_label_down = cv2.resize(pseudo_label, (W_d, H_d), interpolation=cv2.INTER_NEAREST)
        
        # Rearrange dino features to (channels, H, W)
        dino_feat = np.transpose(dino_feat, (2, 0, 1))
        
        # Convert to torch tensors
        dino_feat = torch.from_numpy(dino_feat).float()
        pseudo_label_down = torch.from_numpy(pseudo_label_down).long()
        
        return dino_feat, pseudo_label_down

class SimpleMLP(nn.Module):
    """
    A simple MLP for semantic segmentation implemented as 1x1 convolutions.
    """
    def __init__(self, in_channels=384, num_classes=20):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.fc2 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#############################################
# METRICS & TRAINING/EVALUATION             #
#############################################

def compute_pixel_accuracy(pred, target):
    """Computes pixel accuracy between prediction and ground truth."""
    correct = (pred == target).float().sum()
    total = torch.numel(target)
    return correct / total

def compute_mIoU(pred, target, num_classes=20):
    """Computes mean Intersection-over-Union (mIoU)."""
    ious = []
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    for cls in range(num_classes):
        pred_inds = (pred_np == cls)
        target_inds = (target_np == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1)
            all_preds.append(preds)
            all_targets.append(target)
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    pixel_acc = compute_pixel_accuracy(all_preds, all_targets)
    miou = compute_mIoU(all_preds, all_targets)
    return epoch_loss, pixel_acc, miou

#############################################
# MAIN TRAINING & EVALUATION SCRIPT         #
#############################################

def main():
    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # Prepare dataset file lists
    # -----------------------------
    # In a real scenario, you would iterate over your data directories for the sequences:
    # Training sequences: [00, 01, 02, 03, 04, 05, 06, 07, 09, 10]
    # Testing sequence: [08]
    # Each scan should have paths for:
    #   - dino_features (.npy)
    #   - the generated pseudo-label (.npy), which you can produce using generate_pseudo_label(...)
    #
    # Here we demonstrate with dummy scans. Replace these with your actual file paths.
    train_scans = []
    test_scans = []
    
    # Dummy scan example for training and testing:
    dummy_train_scan = {'dino': 'dummy_dino_train.npy', 'pseudo': 'dummy_pseudo_train.npy'}
    dummy_test_scan  = {'dino': 'dummy_dino_test.npy', 'pseudo': 'dummy_pseudo_test.npy'}
    train_scans.append(dummy_train_scan)
    test_scans.append(dummy_test_scan)
    
    # Create dummy files if they do not exist (for demonstration purposes)
    if not os.path.exists('dummy_dino_train.npy'):
        dummy_dino = np.random.rand(50, 50, 384).astype(np.float32)
        np.save('dummy_dino_train.npy', dummy_dino)
    if not os.path.exists('dummy_pseudo_train.npy'):
        dummy_pseudo = np.random.randint(0, 20, size=(270, 270)).astype(np.uint8)
        np.save('dummy_pseudo_train.npy', dummy_pseudo)
    if not os.path.exists('dummy_dino_test.npy'):
        dummy_dino = np.random.rand(50, 50, 384).astype(np.float32)
        np.save('dummy_dino_test.npy', dummy_dino)
    if not os.path.exists('dummy_pseudo_test.npy'):
        dummy_pseudo = np.random.randint(0, 20, size=(270, 270)).astype(np.uint8)
        np.save('dummy_pseudo_test.npy', dummy_pseudo)
    
    # Create dataset and dataloaders
    train_dataset = SemanticKittiDataset(train_scans)
    test_dataset = SemanticKittiDataset(test_scans)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # -----------------------------
    # Initialize Model, Loss, Optimizer
    # -----------------------------
    model = SimpleMLP(in_channels=384, num_classes=20).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # -----------------------------
    # Training Loop
    # -----------------------------
    num_epochs = 5  # Use more epochs for a real experiment
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, pixel_acc, miou = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Pixel Acc: {pixel_acc:.4f}, mIoU: {miou:.4f}")
    
    # -----------------------------
    # Visualization of Predictions
    # -----------------------------
    # Visualize one training and one testing prediction.
    model.eval()
    cmap = create_semkitti_label_colormap()
    with torch.no_grad():
        # Training prediction
        data, _ = next(iter(train_loader))
        data = data.to(device)
        output = model(data)
        preds = torch.argmax(output, dim=1).cpu().numpy()[0]
        # Upsample prediction to original dummy pseudo-label size (e.g., 270x270)
        preds_up = cv2.resize(preds.astype(np.uint8), (270, 270), interpolation=cv2.INTER_NEAREST)
        pred_vis = cmap[preds_up]
        cv2.imwrite("train_prediction.png", pred_vis)
        
        # Testing prediction
        data, _ = next(iter(test_loader))
        data = data.to(device)
        output = model(data)
        preds = torch.argmax(output, dim=1).cpu().numpy()[0]
        preds_up = cv2.resize(preds.astype(np.uint8), (270, 270), interpolation=cv2.INTER_NEAREST)
        pred_vis = cmap[preds_up]
        cv2.imwrite("test_prediction.png", pred_vis)
    
    # -----------------------------
    # Save Performance Report
    # -----------------------------
    with open("performance_report.txt", "w") as f:
        f.write(f"Final Test Pixel Accuracy: {pixel_acc:.4f}\n")
        f.write(f"Final Test mIoU: {miou:.4f}\n")
    
    print("Training complete. Visualizations and performance report saved.")

if __name__ == "__main__":
    main()
