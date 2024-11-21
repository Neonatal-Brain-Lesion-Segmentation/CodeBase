import SimpleITK as sitk
import matplotlib.pyplot as plt

image_path = "/Users/amograo/Research_Projects/DL_HIE_2024/output/images/hie-lesion-segmentation/MGHNICU_302-VISIT_01_lesion.mha"
orginal_path = "/Users/amograo/Research_Projects/DL_HIE_2024/data_dir/BONBID2023_Val/3LABEL/MGHNICU_302-VISIT_01_lesion.mha"

image = sitk.ReadImage(image_path)
array = sitk.GetArrayFromImage(image)

org_img = sitk.ReadImage(orginal_path)
org_array = sitk.GetArrayFromImage(org_img)

print(array.shape)
print(org_array.shape)

# Plot the Original and Predicted Image (Slices side by side)
for i in range(array.shape[0]):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(org_array[i], cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(array[i], cmap='gray')
    ax[1].set_title('Predicted Image')
    plt.show()