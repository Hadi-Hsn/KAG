# KAG - Knowledge Augmented Generation

AI-powered PDF question answering system with text highlighting capabilities.

---

## 📄 Associated Publication

This project is based on the following peer-reviewed paper:

**KAG: A Scalable Knowledge-Augmented Generation System for Educational Content Management**
Hadi Hasan, Ali Ismail, Ammar Mohanna, and Ali Chehab
*2025 3rd International Conference on Foundation and Large Language Models (FLLM)*, pp. 503–508, 2025.
DOI: [https://doi.org/10.1109/FLLM67465.2025.11391105](https://doi.org/10.1109/FLLM67465.2025.11391105)
IEEE Xplore: [https://ieeexplore.ieee.org/document/11391105](https://ieeexplore.ieee.org/document/11391105)

If you use this repository, system design, or any part of the implementation in academic work, research, or publications, **please cite the above paper**.

### BibTeX Citation

```bibtex
@INPROCEEDINGS{11391105,
  author={Hasan, Hadi and Ismail, Ali and Mohanna, Ammar and Chehab, Ali},
  booktitle={2025 3rd International Conference on Foundation and Large Language Models (FLLM)},
  title={KAG: A Scalable Knowledge-Augmented Generation System for Educational Content Management},
  year={2025},
  pages={503-508},
  keywords={Content management;Accuracy;Adaptive systems;Databases;Large language models;Search methods;Retrieval augmented generation;Microservice architectures;Vectors;Reliability;retrieval-augmented generation;large language models;vector databases;document processing},
  doi={10.1109/FLLM67465.2025.11391105}
}
```

---

## 🛠 Requirements

* Docker & Docker Compose
* OpenAI API Key

---

## 🚀 Quick Start

1. **Set up environment**:

   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

2. **Start the application**:

   ```bash
   docker-compose up --build -d
   ```

3. **Access the web interface**:

   Open `index.html` in your browser

4. **Stop the application**:

   ```bash
   docker-compose down
   ```

---

## 📘 Usage

* Upload PDF documents via the web interface
* Ask questions about your documents
* View highlighted text passages that support the answers

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
