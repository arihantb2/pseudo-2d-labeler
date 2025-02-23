def create_semkitti_label_colormap(): 
    """Creates a label colormap used in SEMANTICKITTI segmentation benchmark.

    Returns:
        A colormap for visualizing segmentation results in BGR format.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [245, 150, 100]       # "car"
    colormap[2] = [245, 230, 100]       # "bicycle"
    colormap[3] = [150, 60, 30]         # "motorcycle"
    colormap[4] = [180, 30, 80]         # "truck"
    colormap[5] = [255, 0, 0]           # "other-vehicle"
    colormap[6] = [30, 30, 255]         # "person"
    colormap[7] = [200, 40, 255]        # "bicyclist"
    colormap[8] = [90, 30, 150]         # "motorcyclist"
    colormap[9] = [255, 0, 255]         # "road"
    colormap[10] = [255, 150, 255]      # "parking"
    colormap[11] = [75, 0, 75]          # "sidewalk"
    colormap[12] = [75, 0, 175]         # "other-ground"
    colormap[13] = [0, 200, 255]        # "building"
    colormap[14] = [50, 120, 255]       # "fence"
    colormap[15] = [0, 175, 0]          # "vegetation"
    colormap[16] = [0, 60, 135]         # "trunk"
    colormap[17] = [80, 240, 150]       # "terrain"
    colormap[18] = [150, 240, 255]      # "pole"
    colormap[19] = [0, 0, 255]          # "traffic-sign"
    return colormap