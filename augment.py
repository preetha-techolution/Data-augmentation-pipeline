import os
import cv2
import json
import pandas as pd
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

def process_images(input_csv_path, images_folder_path, output_folder_path, num_augmentations=5):
    """
    Processes images by applying different augmentation techniques separately,
    generating multiple augmented images and corresponding annotations for each technique.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    df = pd.read_csv(input_csv_path)
    augmented_rows = []  # Store augmented rows for output CSV
    
    # Define augmentation techniques
    augmenters = {
        #"Multiply": iaa.Multiply((1.2, 1.5)),
        #"Affine": iaa.Affine(rotate=10, scale=(0.5, 0.7)),
        "Rotate" : iaa.Rotate((-90, 90)),
        "Noise": iaa.Add((-40, 40)),
        "Brightness": iaa.WithBrightnessChannels(
    iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV]),
        "Hue" : iaa.MultiplyHue((0.5, 1.5)),
        "Saturation" : iaa.MultiplySaturation((0.5, 1.5)),
         "Contrast" : iaa.GammaContrast((0.5, 2.0)),
        "Fliplr" : iaa.Fliplr(0.5),
        "Flipud" : iaa.Flipud(0.5),
        "Shear" : iaa.Affine(shear=(-20, 20)),
        "Translate" : iaa.TranslateX(px=(-90, 90)),
        "Sharpness" : iaa.pillike.EnhanceSharpness()
        }
    
    ia.seed(1)  # Set a fixed seed for reproducibility

    for idx, row in df.iterrows():
        image_name = row['name']
        image_path = os.path.join(images_folder_path, image_name)

        if not os.path.isfile(image_path):
            print(f"Image {image_name} not found in the specified folder. Skipping.")
            continue
        
        try:
            annotations = json.loads(row['imageAnnotations'].replace('\"', '"'))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for image {image_name}: {e}. Skipping this image.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_name}. Skipping.")
            continue

        # Collect all keypoints from all annotations
        all_keypoints = []
        for annotation in annotations:
            vertices = annotation['vertices']
            for vertex in vertices:
                all_keypoints.append(Keypoint(x=int(vertex['x']), y=int(vertex['y'])))

        kps = KeypointsOnImage(all_keypoints, shape=image.shape)

        # Apply each augmentation technique separately
        for augmenter_name, augmenter in augmenters.items():
            for i in range(num_augmentations):
                image_aug, kps_aug = augmenter(image=image, keypoints=kps)

                # Process augmented keypoints back into annotations
                keypoint_idx = 0
                modified_annotations = []
                
                for annotation in annotations:
                    vertices = annotation['vertices']
                    modified_vertices = []
                    
                    for j in range(4):  # Assuming 4 keypoints per annotation
                        augmented_kp = kps_aug.keypoints[keypoint_idx]
                        modified_vertex = vertices[j].copy()
                        modified_vertex['x'] = int(augmented_kp.x)
                        modified_vertex['y'] = int(augmented_kp.y)
                        modified_vertices.append(modified_vertex)
                        keypoint_idx += 1
                    
                    modified_annotation = annotation.copy()
                    modified_annotation['vertices'] = modified_vertices
                    modified_annotations.append(modified_annotation)

                # Update and save augmented annotations
                augmented_row = row.copy()
                augmented_row['imageAnnotations'] = json.dumps(modified_annotations)
                augmented_row['name'] = f"{os.path.splitext(image_name)[0]}_{augmenter_name}_{i+1}.jpg"
                augmented_rows.append(augmented_row)

                # Save augmented image
                output_image_path = os.path.join(output_folder_path, augmented_row['name'])
                image_after = kps_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
                cv2.imwrite(output_image_path, image_after)

    # Save augmented annotations to CSV
    augmented_df = pd.DataFrame(augmented_rows)
    output_csv_path = 'augmented_annotations.csv'
    augmented_df.to_csv(output_csv_path, index=False)
    print(f"Processing complete. Modified annotations saved to '{output_csv_path}'")

def main():
    input_csv_path = 'dataSetCollection_train-phase-1-14_resources.csv'
    images_folder_path = 'unique_images'
    output_folder_path = 'augmented_images'
    num_augmentations = 5  # Number of augmentations per technique
    
    process_images(input_csv_path, images_folder_path, output_folder_path, num_augmentations)

if __name__ == "__main__":
    main()
