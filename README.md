<a id="readme-top"></a>

![Contributors][contributors-shield]
![Forks][forks-shield]
![Stars][stars-shield]
![Issues][issues-shield]
![License][license-shield]

<div align="center">
    <img src="img/dama-logo.png" alt="Logo" width=150px>
    <h3 align="center"> DAMA - Braille Image-to-Text Converter  </h3>
    <p align="center">
        Turns photos of Filipino/English Grade 1 Braille into plain text!
    </p>
</div>

<details>
    <summary> Table of Contents</summary>
        <ol>
            <li> <a href="#about-dama"> About DAMA </a> </li>
            <li> <a href="#getting-started"> Getting Started </a> 
                <ul> 
                    <li> <a href="#installation-and-setup"> Installation and Setup </a> 
                </ul>
            </li>
            <li> <a href="#usage"> Usage </a> </li>
            <li> <a href="#screenshots"> Screenshots </a> </li>
            <li> <a href="#license"> License </a> </li>
            <li> <a href="#developers"> Developers </a> </li>
        </ol>
</details>

## About DAMA

<div align="center">
    <img src="img/dama-banner.png" alt="Dama banner"/>
</div>

An Optical Character Recognition (OCR) project that detects Filipino/English Grade 1 Braille characters using YOLO and converts them into text with the help of LLMs.

## Getting Started

This is an example of how you may give instructions on setting up your project locally. To get a local copy up and running follow these simple example steps.

### Prerequisities

- Python 3.11 or higher
- [Groq API Key](https://console.groq.com/keys)

### Installation and Setup

1. Clone the repository

   ```bash
   git clone https://github.com/khat001/DAMA-Filipino-Braille-Converter
   cd DAMA-Filipino-Braille-Converter
   ```

2. Setup python virtual environment

   ```bash
   python -m venv .venv
   ```

3. Activate virtual environment

   ```bash
   source .venv/bin/activate  # Linux
   .venv/Scripts/activate # Windows
   ```

4. Install requirements

   ```bash
   pip install -r requirement.txt
   ```

5. Create a .env file and insert your GROQ API
   ```
   GROQ_API = "gsk_1234567"
   ```

## Usage

This section shows how to use the actual app.

1. Go to `braille_app`

   ```bash
   cd braille_app
   ```

2. Download the required models

   ```bash
   python -m scripts.download_model
   ```

3. Run the app and click the `http` server
   ```bash
   flask run
   ```

**It should now take you to the home page.**

![Home Page][screenshot-index]

<p align="center"> Screenshot of Home Page</p>

## Screenshots

<h3 align="center"> Sample Detection and Conversion </h3>

![Screenshot of Detection][screenshot-sample-detection]

<h3 align="center"> History Records </h3>

![Screenshot of History Records][screenshot-history]

## License

Distributed under MIT License. See `LICENSE` for more information.

## Developers

<a href="https://github.com/khat001/DAMA-Filipino-Braille-Converter/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=khat001/DAMA-Filipino-Braille-Converter" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[screenshot-index]: img/screenshot-index.png
[screenshot-sample-detection]: img/screenshot-sample-detection.png
[screenshot-history]: img/screenshot-history.png
[contributors-shield]: https://img.shields.io/github/contributors/khat001/DAMA-Filipino-Braille-Converter?style=for-the-badge&color
[forks-shield]: https://img.shields.io/github/forks/khat001/DAMA-Filipino-Braille-Converter?style=for-the-badge
[stars-shield]: https://img.shields.io/github/stars/khat001/DAMA-Filipino-Braille-Converter?style=for-the-badge
[issues-shield]: https://img.shields.io/github/issues/khat001/DAMA-Filipino-Braille-Converter?style=for-the-badge
[license-shield]: https://img.shields.io/github/license/khat001/DAMA-Filipino-Braille-Converter?style=for-the-badge
