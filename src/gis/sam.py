from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np

def mask_gen(img):
    """
    Returns a list of dictionraries. each dictionary holds:
    - segmentation ; (nxn) mask
    - area ; size of segmentation
    - bbox
    - predicted iou
    - point coords
    - stability score
    - crop box 

    """
    mask_generator = load_sam()
    masks = mask_generator.generate(img)
    return masks 

def load_sam():
    SAM_CHECKPOINT = "/home/fergus/sam.pth"
    model_type = 'vit_h'
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def load_sam_predictor():
    SAM_CHECKPOINT = "/home/fergus/sam.pth"
    model_type = 'vit_h'
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    return predictor


def predict_with_point(predictor, image, point, label=1):
    """
    predictor: SamPredictor instance
    image: numpy array (H, W, 3) in RGB
    point: (x, y) coordinates of user click (pixel coords)
    label: 1 = positive point (object), 0 = negative point (background)
    """
    predictor.set_image(image)

    input_point = np.array([point])
    input_label = np.array([label])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,  # returns multiple mask candidates
    )
    return masks, scores, logits
