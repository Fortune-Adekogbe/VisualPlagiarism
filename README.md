# Visual Plagiarism Detection Using Machine Learning

## Project Overview

This repository contains the code and resources for a project on visual plagiarism detection using machine learning. The project aims to identify cases where graphic design or user interface images may have been plagiarized.

## Project Structure

### Code
- **Programming Language:** Python
- **Libraries Used:** TensorFlow, FAISS, and other model-specific packages.

### Notebooks
1. **Visual Plagiarism Notebook:** 
   - This notebook details a series of experiments with various language-image pre-trained models, including the CLIP, MetaCLIP, MobileCLIP, SigLIP, and BLIP models.
   - The models are used to extract embeddings from images, which capture the visual context of the image.
   - Similarity metrics such as Cosine Similarity, Inner Product, and L2 Distance are used to compute the difference between embeddings.
   - The FAISS package is utilized for efficient similarity search and indexing, with all indexes stored in the `data` folder.

2. **Data Preparation Notebook:**
   - This notebook explains the process of extracting a subset of the Common Screens dataset.
   - The dataset originally contained over 80 million samples, but only a subset of less than 200,000 images was downloaded for this project.
   - The downloaded images were cleaned by removing duplicates, blank images, and explicit content.
   - Images were split into training, testing, and validation sets.
   - Images were then prepared for further analysis by pairing them and storing them in HDF5 files. Embeddings were extracted from the images using the CLIP and Meta CLIP models, and these embeddings were also stored in HDF5 files.

3. **Custom Embedding Evaluation Notebooks:**
   - The HDF5 files for the CLIP and MetaCLIP embedding models were loaded into memory for model training.
   - Custom embedding evaluators were developed using dense neural networks and a 1-D convolutional neural network, implemented with TensorFlow.
   - The resulting models were evaluated and compared with an approach involving the use of standard evaluation metrics.
   - The standard evaluation metrics outperformed the custom embedding models.

## Usage

1. **Setup:**
   - Ensure you have Python and the necessary libraries installed. The primary libraries include TensorFlow and FAISS.

2. **Data Preparation:**
   - Use the Data Preparation Notebook to download, clean, and prepare the dataset. Extract embeddings and store them in HDF5 files.

3. **Model Training and Experimentation:**
   - Train the evaluator models using the Custom Embedding Evaluator notebooks. Experiment with these evaluators and the different similarity metrics and models to achieve the best performance.

4. **Evaluation:**
   - Evaluate the models on an evaluation set. Compare the performance of custom models against standard evaluation metrics.

## Results

- The top-performing model was the CLIP model, which achieved the best results in terms of Top-1, Top-3, and Top-5 accuracy, as well as latency.

## Data

- **Dataset:**
  - A subset of the Common Screens dataset.
  - An evaluation set collected during the course of this research.
- **Size:** Approximately 200,000 images.
- **Storage:** Images and embeddings are stored in HDF5 files.

## Conclusion

This project demonstrates a machine learning approach to detecting visual plagiarism by leveraging advanced models like CLIP and Meta CLIP. The results indicate that standard evaluation metrics can outperform custom embedding models in certain scenarios.

## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/mit) file for details.

## Acknowledgments

Special thanks to the developers of TensorFlow, FAISS, and the creators of the LIP models for providing the foundational tools used in this project.
