# Cognify:  English to Hindi Translation using mBART and OPUS NLLB Dataset

![image](https://github.com/user-attachments/assets/d400b503-9c9c-4ebb-a6c6-cab525450d88)
This project fine-tunes the **mBART-large-50** multilingual translation model on the English-Hindi parallel corpus from the [OPUS NLLB dataset](https://object.pouta.csc.fi/OPUS-NLLB/v1/xml/en-hi.xml.gz) to create a context-aware English→Hindi translation system.

## Overview

This project delivers a comprehensive AI-powered system that translates English text into various Indian regional languages with **high accuracy, minimal human intervention**, and **context-aware interpretation** tailored to different app domains (navigation, restaurant discovery, travel, etc.).

The system is designed to handle **handwritten and printed English text inputs**, extract relevant information, and provide actionable insights in the target language, enhancing user experience across diverse applications.

---

## Core Features

### 1. Robust Multilingual Translation

- Fine-tuned **mBART-large-50** Transformer model for high-quality English → regional language translation (starting with Hindi).  
- Maintains **context, tone, and semantic accuracy** to minimize errors.  
- Supports extension to multiple Indian languages (Marathi, Tamil, Telugu, Bengali, etc.).

### 2. OCR Integration for Text Input

- Supports **image uploads** of handwritten or printed English text.  
- Uses OCR (e.g., Tesseract) to accurately extract English text from images before translation.  
- Enables natural input beyond typed text, making the system versatile for real-world usage.

### 3. Entity Extraction and Information Parsing

- Applies **NLP pipelines** to extract key entities like **locations, dates, hotel/restaurant names**, and other relevant data from translated text.  
- Enables downstream app logic to interpret user requests properly based on context.

### 4. Domain-Specific API Integration

- For **Navigation apps:**  
  - Translates source and destination locations into the local language.  
  - Integrates with **Google Maps API** to fetch the shortest route, travel distance, and directions.  
  - Presents route info fully in the regional language.

- For **Restaurant Discovery apps:**  
  - Extracts hotel/restaurant names and ratings.  
  - Compares and recommends the best option in the user’s preferred regional language.  
  - Optionally integrates with APIs like Zomato or Google Places.

### 5. Error Handling and Fallback Mechanisms

- Detects **ambiguous translations or OCR errors**.  
- Provides user-friendly fallback options or clarifying prompts.  
- Ensures system robustness in noisy real-world scenarios.

### 6. Low Latency & Scalability

- Optimized model inference for **near real-time translations**.  
- Modular architecture designed for scaling to multiple languages and increasing user load.

### 7. User-Friendly Frontend & AI Chatbot

- Intuitive **frontend interface** for easy interaction, including text input and image upload.  
- Integrated **AI chatbot** that offers smarter suggestions, clarifications, and conversational support.  
- Enhances accessibility for non-technical users.

### 8. Ethical & Privacy Compliance

- All data handling follows **privacy regulations and ethical guidelines**.  
- No user data is stored without consent; sensitive information is handled securely.

---

## Project Structure

- `data/` — Dataset files (parallel corpora).  
- `scripts/` — Python scripts for parsing dataset, training mBART, inference, evaluation, OCR integration, entity extraction, and API calls.  
- `models/` — Saved fine-tuned models.  
- `frontend/` — Web or mobile UI code (if implemented).  
- `notebooks/` — Exploratory analysis and prototyping.  
- `README.md` — This file.  
- `requirements.txt` — Dependencies.

---

## How to Use

1. **Prepare Dataset:** Download and parse OPUS NLLB or similar parallel corpus.  
2. **Train Model:** Fine-tune mBART on your dataset for accurate translation.  
3. **Integrate OCR:** Use the OCR module to extract English text from images.  
4. **Run Entity Extraction:** Parse translated text to identify locations, dates, or names.  
5. **Call APIs:** Query Google Maps or other relevant APIs for app-specific data.  
6. **Frontend:** Present results with seamless user interaction and chatbot assistance.

---

## Future Enhancements

- Expand support for more Indian regional languages.  
- Improve OCR accuracy with custom models for local handwriting styles.  
- Add voice input and speech-to-text modules.  
- Implement real-time collaborative translation features.

---

## References

- [Facebook mBART](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)  
- [OPUS NLLB Dataset](https://object.pouta.csc.fi/OPUS-NLLB/v1/xml/en-hi.xml.gz)  
- [Google Maps API](https://developers.google.com/maps/documentation)  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)  

---

## License

Specify your preferred license here (e.g., MIT, Apache 2.0).

---

## Contact

For questions or collaboration, reach out at: your.email@example.com
