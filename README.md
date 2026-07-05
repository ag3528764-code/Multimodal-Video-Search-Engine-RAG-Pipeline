# 🎥 Multimodal Video Search Engine & RAG Pipeline

A lightweight, high-performance Multimodal Retrieval-Augmented Generation (RAG) system built from scratch using **PyTorch**, **CLIP**, and **Whisper**. This application processes raw video files (`.mp4`), extracts and aligns both visual frames and spoken acoustic signals into a shared vector space, and allows users to search for precise moments in a video using natural language queries.

---

## 🏗️ System Architecture

The pipeline avoids heavy, bloated database infrastructures by implementing an in-memory tensor-based vector index utilizing raw mathematical dot-product similarity configurations.
