# 🩺 Skin Disease Classification System

_✨ Unlock the power of AI-driven dermatology with this web-based tool! Instantly analyze images and learn about common skin conditions—all from the convenience of your browser._

## 🚀 What Does This App Do?

- **Instant Predictions**  
  Upload a photo of a skin condition and let our smart AI model tell you what it might be—**in seconds**! Each result comes with a confidence score and an easy-to-understand explanation.

- **8 Disease Classes Covered**  
  - **Bacterial**: Cellulitis, Impetigo  
  - **Fungal**: Athlete’s foot, Nail fungus, Ringworm  
  - **Parasitic**: Cutaneous larva migrans  
  - **Viral**: Chickenpox, Shingles

- **Learn As You Explore**  
  Read detailed descriptions for each prediction and empower yourself with knowledge. The app is crafted to **educate and inform**—perfect for students, educators, and the health-curious.

- **Sleek, Modern Interface**  
  Designed with Bootstrap for a seamless and **mobile-friendly** experience. Interact easily from any device.

## 🖼️ Your Path to Prediction

1. **Clone & Set Up**
    ```bash
    git clone 
    cd skin-disease
    pip install -r requirements.txt
    ```

2. **Make Sure You’ve Got the Model**
   Place `skindiseasemodel.h5` in the project root.

3. **Start Your Journey**
    ```bash
    python APP.PY
    ```
    Visit [http://localhost:5000](http://localhost:5000) in your web browser.

4. **The Magic Moment**
   - Hit “Choose File” to select an image.
   - Click “Predict”.
   - Voilà! See your result, confidence score, and a brief description.

## 🛠️ What’s Inside?

```
skin-disease/
├── APP.PY                 # Your gateway: The Flask app
├── skindiseasemodel.h5    # The AI brain
├── skindisease.ipynb      # Model training notebook
├── requirements.txt       # Everything you need, in one file
├── templates/             # Beautiful HTML pages
│   ├── index.html
│   ├── about.html
│   ├── project.html
│   └── contact.html
└── skin-disease-datasaet/
    └── images/            # The learning set
```

## ⚙️ Under the Hood: AI Model Specs

- **Input:** 224x224 RGB images
- **CNN Layers:** 3 convolutional layers with max pooling
- **Dense Layer:** 128 units (ReLU)
- **Output:** 8-way softmax
- **Optimizer:** Adam
- **Loss:** Sparse categorical crossentropy
- **Training:** 10 epochs of data-driven learning

## 🌐 API Endpoints at a Glance

- `GET /` — Main prediction page
- `GET /about` — About this project
- `GET /project` — Deep dive into details
- `GET /contact` — Get in touch!
- `POST /predict` — Power your prediction
- `GET /api/classes` — Check out all disease classes

## 👩⚕️ A Friendly Reminder
> _This application is **for educational and research purposes only**. It’s not a substitute for professional medical advice, diagnosis, or treatment. When it comes to your health, always consult a qualified healthcare professional!_

## 👥 Contributors Welcome!

Love code? Passionate about healthcare?  
**Jump in!** Report bugs, request features, or submit your PR.

## 📧 Contact

- **Lead:** Ankur Saini
- **Email:** sainianku018@gmail.com

## 🙏 Special Thanks

- Big thanks to the open dataset contributors, the incredible TensorFlow/Keras community, and the medical professionals who inspired this project!

_A project with heart, for minds that are curious._  
_Licensed under MIT. Explore. Learn. Contribute!_
