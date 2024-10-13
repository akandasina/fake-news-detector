## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - nltk

## How to Run
1. Clone or download this repository.
2. Install the required libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn nltk

## NLTK Setup
Before running the script, you need to download the NLTK stopwords. Run the following command in your Python environment:

import nltk
nltk.download('stopwords')

## Dataset
The dataset consists of articles with the following attributes:
- id: Unique ID for a news article
- title: The title of a news article
- author: Author of the news article
- text: The text of the article
- label: A label that marks the article as potentially unreliable (1: unreliable, 0: reliable)

## Author
This project was created by Akanda Sina Kilicarslan as a self-directed initiative in my pursuit of knowledge and skills in the field of computer science.

## License
This project is licensed under the MIT License - see the LICENSE file for details.