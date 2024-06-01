function Home() {
    return(
        <div class="home">
            <div class="intro">
                <p>
                  This project is a college initiative aimed at identifying defects on metal surfaces and metal shafts. We use a Convolutional Neural Network (CNN) model, which is a type of deep learning model particularly effective for image analysis tasks. 
                  The model is trained on a public dataset from Kaggle, which contains various images of metal surfaces and shafts, both with and without defects. 
                  The CNN model, learns to identify patterns and features in these images that indicate the presence of a defect. 
                  This allows us to input a new, unseen image of a metal surface or shaft, and the model will output a prediction of whether a defect is present.
                </p>
            </div>
        </div>
    );
}

export default Home;