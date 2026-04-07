import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import cv2
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor


def stitch(patches, boxes, patch_per_img):
    # patches : (num_images * patch_per_img, channels, patch_size, patch_size)
    # boxes : contains (y1, y2, x1, x2) for each patch
    num_patches = patches.shape[0]
    assert num_patches % patch_per_img == 0
    num_images = num_patches // patch_per_img

    device = patches.device
    dtype = patches.dtype
    C = patches.shape[1]

    all_images = []

    for b in range(num_images):
        # Extract the bounding boxes for the current image
        img_boxes = boxes[b * patch_per_img: (b + 1) * patch_per_img]

        # Reconstruct the original full image sizes from the bounding boxes
        H = max(box[1] for box in img_boxes)
        W = max(box[3] for box in img_boxes)

        # Create accumulators for the blended image and the blending weights
        img_acc = torch.zeros((C, H, W), device=device, dtype=torch.float32)
        weight_acc = torch.zeros((1, H, W), device=device, dtype=torch.float32)

        for p_idx in range(patch_per_img):
            global_p_idx = b * patch_per_img + p_idx
            y1, y2, x1, x2 = img_boxes[p_idx]
            h_box, w_box = y2 - y1, x2 - x1

            # Resize patch back to its original bounding box size
            patch = patches[global_p_idx: global_p_idx + 1]
            resized_patch = F.interpolate(patch, size=(h_box, w_box), mode='bicubic', align_corners=False).squeeze(0)

            # Create a 2D cosine/hann window (peaks at 1 in the middle, decays to 0 at the edges)
            wy = torch.cos(torch.linspace(-math.pi / 2, math.pi / 2, h_box, device=device))
            wx = torch.cos(torch.linspace(-math.pi / 2, math.pi / 2, w_box, device=device))
            window = torch.ger(wy, wx).unsqueeze(0) + 1e-5  # Add a tiny epsilon to prevent division by zero

            # Accumulate the weighted patch and the weights
            img_acc[:, y1:y2, x1:x2] += resized_patch * window
            weight_acc[:, y1:y2, x1:x2] += window

        # Normalize the accumulated image by the sum of weights (this performs the smooth blending)
        reconstructed = img_acc / weight_acc
        all_images.append(reconstructed.to(dtype))

    return all_images


def process_and_save(image, mask_idx, name, output_dir, cmap, semaphore):
    """Worker function executed in parallel by the thread pool."""
    try:
        # Processing
        mask_rgba = cmap[mask_idx]

        mask_a = mask_rgba[:, :, 3:].astype(np.float32) / 255.0
        mask_bgr = mask_rgba[:, :, 2::-1].astype(np.float32)

        image_bgr = image[:, :, ::-1].astype(np.float32)
        overlay = image_bgr * (1 - mask_a * 0.5) + mask_bgr * (mask_a * 0.5)

        mask_bgr = mask_bgr.astype(np.uint8)
        overlay = overlay.clip(0, 255).astype(np.uint8)

        # Saving
        write_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        cv2.imwrite(str(output_dir / f'{name}_mask.png'), mask_bgr, write_params)
        cv2.imwrite(str(output_dir / f'{name}_over.png'), overlay, write_params)
    finally:
        # Always release the semaphore, even if an exception occurs
        semaphore.release()


def mask_to_xanylabelling_json(mask, class_mapping, image_name, output_dir, semaphore,
                               approx_epsilon_factor=0.002, min_polygon_area=250.0, edge_margin=5):
    """
    Converts a segmentation mask to an X-AnyLabelling compatible JSON file.

    :param mask: 2D numpy array of shape (H, W) containing class indices.
    :param class_mapping: Dictionary mapping class names (str) to integers (mask values).
    :param image_name: Name of the image.
    :param output_dir: Directory where the output JSON will be saved.
    :param semaphore:
    :param approx_epsilon_factor: Factor for contour approximation. Set to 0 to keep all points.
    :param min_polygon_area: Minimal area per polygon
    :param edge_margin: Number of pixels from the image boundary to ignore (set to 0)
    """
    try:
        H, W = mask.shape

        # Base structure of the JSON expected by X-AnyLabelling
        data = {
            "version": "3.3.10",
            "flags": {
                "souple": False,
                "rigide": False
            },
            "shapes": [],
            "imagePath": f'{image_name}.jpg',
            "imageData": None,
            "imageHeight": int(H),
            "imageWidth": int(W)
        }

        # Iterate over the specified classes
        for class_name, class_index in class_mapping.items():
            # Skip the background class
            if class_index == 0:
                continue

            # Create a binary mask for the current class (cv2 requires uint8)
            binary_mask = np.uint8(mask == class_index)

            if edge_margin > 0 and H > 2 * edge_margin and W > 2 * edge_margin:
                binary_mask[:edge_margin, :] = 0  # Top edge
                binary_mask[-edge_margin:, :] = 0  # Bottom edge
                binary_mask[:, :edge_margin] = 0  # Left edge
                binary_mask[:, -edge_margin:] = 0  # Right edge

            # Find contours
            # RETR_EXTERNAL extracts only the outer boundaries. If you have holes in your masks
            # that need annotating, change to cv2.RETR_TREE or cv2.RETR_CCOMP.
            # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments.
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Skip noise (e.g., predicted blobs smaller than 10 pixels in area)
                if cv2.contourArea(contour) < min_polygon_area:
                    continue

                # Smooth/approximate the polygon to reduce points.
                # This makes manual editing in X-AnyLabelling much easier.
                if approx_epsilon_factor > 0:
                    epsilon = approx_epsilon_factor * cv2.arcLength(contour, True)
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                # Reshape contour from (N, 1, 2) to (N, 2) and convert to nested lists
                # We convert to float to ensure json serialization works correctly
                points = contour.reshape(-1, 2).astype(float).tolist()

                # A valid polygon needs at least 3 points
                if len(points) < 3:
                    continue

                # Construct the shape dictionary
                shape = {
                    "label": class_name,
                    "score": None,
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "difficult": False,
                    "shape_type": "polygon",
                    "flags": {},
                    "attributes": {},
                    "kie_linking": []
                }

                data["shapes"].append(shape)

        # Write the assembled data to the JSON file
        with open(output_dir / f'{image_name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    finally:
        # Always release the semaphore, even if an exception occurs
        semaphore.release()


class CBCSeg(pl.LightningModule):
    def __init__(
            self,
            lr=3e-4,
            weight_decay=1e-4,
            eta_min=1e-6,
            patch_per_img=21,
            save_json=False,
            approx_epsilon_factor=0.002,
            min_polygon_area=250.0,
            edge_margin=5,
            inference_workers=32,
            semaphore_lim=256,
            output_dir=None,
            cmap=None,
            class_mapping=None
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.patch_per_img = patch_per_img
        self.save_json = save_json
        self.approx_epsilon_factor = approx_epsilon_factor
        self.min_polygon_area = min_polygon_area
        self.edge_margin = edge_margin
        self.inference_workers = inference_workers
        self.semaphore_lim = semaphore_lim
        self.output_dir = output_dir
        self.cmap = cmap
        self.class_mapping = class_mapping
        self.executor = None
        self.task_semaphore = None

    def forward(self, x):
        return x

    def on_predict_start(self):
        # Thread pool to handle concurrent processing and saving
        self.executor = ThreadPoolExecutor(max_workers=self.inference_workers)
        # Semaphore limits pending tasks to prevent RAM explosion (Backpressure)
        self.task_semaphore = threading.Semaphore(self.semaphore_lim)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, names, patches, boxes = batch

        # Forward pass
        outputs = self(patches)

        # Stitching
        logits = stitch(outputs, boxes, self.patch_per_img)

        # Processing & Saving
        for b in range(len(images)):
            mask_idx = torch.argmax(logits[b], dim=0).cpu().numpy()
            # Acquire semaphore (will block if too many tasks are already pending)
            self.task_semaphore.acquire()
            # Submit task to the worker pool
            if self.save_json:
                self.executor.submit(
                    mask_to_xanylabelling_json,
                    mask_idx, self.class_mapping, names[b], self.output_dir, self.task_semaphore,
                    self.approx_epsilon_factor, self.min_polygon_area, self.edge_margin
                )
            else:
                self.executor.submit(
                    process_and_save,
                    images[b], mask_idx, names[b], self.output_dir, self.cmap, self.task_semaphore
                )

        return 0

    def on_predict_end(self):
        # Wait for all remaining background tasks to finish when prediction completes
        self.executor.shutdown(wait=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Smoothly decreases the learning rate to lr_min over max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.eta_min  # minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Update the learning rate after every epoch
                "frequency": 1
            },
        }
