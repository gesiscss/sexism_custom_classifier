# Changelog

## [0.0.10] - 2020-12-07

### Added

- Model : CNN model with pre-trained embeddings
- Model : Baseline models (gender word and toxicity)

## [0.0.9] - 2020-11-27

### Added

- Model : CNN model with emebedding layer
- Feature : BERT document embeddings preprocessing and feature extraction

## [0.0.8] - 2020-11-12

### Added

- Experiments : Save the classification performance and the parameters of the model.

## [0.0.7] - 2020-10-29

### Added

- Experiments : Balance classes for training and testing (with downsampling)

### Changed

- Run pipeline

## [0.0.6] - 2020-09-30

### Added

- Test data domain preparation
- Feature selection for ngram and type dependency feature sets
- The hyper-parameters tuning with GridSearchCV
- Logistic Regression model

### Changed

- Training data domain preparation
- Run pipeline

## [0.0.5] - 2020-09-18

### Added

- Type dependency preprocessing and feature extraction
- Training data domain preparation

### Changed

- Simplified notebooks.


## [0.0.4] - 2020-09-14

### Added

- CHANGELOG.md file

### Changed

- README.md file

## [0.0.3] - 2020-09-11

### Changed

- Attributes added in setup.py file to run the project on GESIS notebooks.

## [0.0.2] - 2020-09-10

### Added

- Create sklearn pipeline with transformer and featureunion.
- setup.py file
- requirements.txt file
- .gitignore file

### Changed

- Abstract classes for preprocessing and feature extraction steps.
- Neutral and Compound scores use for sentiment feature extraction.

## [0.0.1] - 2020-09-02

### Added

- Preprocessing step for Sentiment and Ngram features.
- Feature Extraction step for Sentiment and Ngram features.

[unreleased]: https://github.com/gesiscss/sexism_custom_classifier/commits/master
[0.0.4]: https://github.com/gesiscss/sexism_custom_classifier/commits/master
[0.0.3]: https://github.com/gesiscss/sexism_custom_classifier/commit/41e22cd41e15d1b377c09b59c7f42a04ec806cc0
[0.0.2]: https://github.com/gesiscss/sexism_custom_classifier/commit/b995507a9fec2fae3005f7b46ac4664f147814d1
[0.0.1]: https://github.com/gesiscss/sexism_custom_classifier/commit/56f1d8540c01b18dc69118db3d406938fa952195