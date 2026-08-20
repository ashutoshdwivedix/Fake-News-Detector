# Fake News Detector

A simple machine learning project that tries to figure out whether a news article is real or fake, based on its text content. Built this mostly to get hands-on practice with NLP and classification models, and to see how far a "traditional" ML approach (as opposed to deep learning) can get on this kind of problem.

It comes with a trained model and a small web app so you can actually test it on your own headlines/articles instead of just looking at accuracy numbers in a notebook.

## How it works

The pipeline is pretty standard for text classification:

1. Clean and preprocess the article text (lowercasing, removing punctuation/stopwords, etc.)
2. Convert the text into numerical features using vectorization (TF-IDF)
3. Feed those features into a classifier trained to label articles as **real** or **fake**
4. Save the trained model so it can be reused without retraining every time

All of the training/experimentation is in the Jupyter notebook (`Fake_News_Detection_using_Machine_Learning...ipynb`), and the final trained model is saved as `fake_news_model.pkl` so the web app can load it directly.

## Project structure

```
├── Fake_News_Detection_using_Machine_Learning.ipynb   # data cleaning, training, evaluation
├── fake_news_model.pkl                                 # trained model, ready to use
├── web.py                                               # simple web app to test the model
└── README.md
```

## Running it locally

Clone the repo and install whatever dependencies `web.py` needs (pandas, scikit-learn, and a web framework like Streamlit):

```bash
git clone https://github.com/ashutoshdwivedix/Fake-News-Detector
cd Fake-News-Detector
pip install pandas scikit-learn streamlit   # if you have one, otherwise install packages manually
python web.py
```

Then open the app in your browser, paste in a news article or headline, and it'll tell you whether it thinks it's real or fake.

## Notes / limitations

This is a learning project, not a production-grade fact-checker. A few honest caveats:

- It's trained on a specific dataset, so it'll work best on articles that look similar in style/topic to what it was trained on
- It's judging *writing style and patterns*, not actually verifying facts — so don't treat the output as a real fact-check
- Accuracy will vary depending on the source and topic of the article you test

## Author

Made by [Ashutosh Dwivedi](https://github.com/ashutoshdwivedix)

Feel free to open an issue or PR if you spot something broken or have ideas to improve it.
