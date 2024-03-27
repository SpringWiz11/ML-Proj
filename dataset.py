from prepare_dataset import prepare_dataset

# Define the directory containing your dataset
data_dir = "/home/kishan/Documents/projects/Lung_disease_prediction/dataset"

# Prepare the dataset
train_loader, test_loader, classes = prepare_dataset(data_dir)

# Now you can use train_loader, test_loader, and classes in your training code
