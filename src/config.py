import torch

# Paths
INPUT_FOLDER  = "input_images"
OUTPUT_FOLDER = "output"

# Device
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Detection
PERSON_DETECTION_CONF = 0.25
SEGMENTATION_CONF     = 0.15
FACIAL_KEYPOINT_CONF  = 0.9

# Visibility
VISIBILITY_THRESHOLD = 0.40
DEEPFACE_BOOST_MAX   = 0.3

# DeepFace / head geometry
DEEPFACE_MIN_CONFIDENCE = 0.4
FACE_EXTENSION          = 0.4
BBOX_CUTOFF_RATIO       = 0.28
UPSCALE_THRESHOLD       = 256
TARGET_UPSCALE_SIZE     = 512

# Mode
ANONYMIZE_FULL_BODY = False

# Inpainting
INPAINT_STRENGTH    = 0.45
GUIDANCE_SCALE      = 7.0
NUM_INFERENCE_STEPS = 50
CROP_PADDING        = 0.1

POSITIVE_PROMPT = "photograph, real person, natural skin"
NEGATIVE_PROMPT = (
    "painting, drawing, illustration, cartoon, anime, cgi, 3d render, "
    "artificial, fake, plastic, doll, unrealistic, deformed, blurry, watermark"
)

SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
