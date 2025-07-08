# Enron Email Search API

A Flask-based hybrid lexical + semantic search API over the Enron email corpus,
with on-demand extractive summaries and lightweight topic clustering.

## Features

- **Ingestion & Threading**  
  Parses raw email files into SQLite, threading by `In-Reply-To` and normalized subjects.
- **Hybrid Search**  
  - **Semantic**: cosine similarity over sentence-transformer embeddings  
- **On-Demand Summaries & Categories**  
  Extractive summaries and mini-KMeans categorization applied only to search results  
- **CORS-Enabled API**  
  `GET /search?q=<your query>` returns JSON, callable from any frontend.

## Prerequisites

- Python 3.8+  
- Git  

## Installation

```bash
# Clone the repo
git clone https://github.com/tajnoor7/HMI
cd backend

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run
1. First add all of the enron files into /backend/INEnron/
2. Then run python main.py


# Angular + Tailwind CSS Starter Setup

This guide sets up an Angular project with **Tailwind CSS** — a modern utility-first CSS framework — for building sleek, responsive UIs with maximum productivity.

---

## Prerequisites

Ensure you have the following installed:

- **Node.js** (v16+ recommended) — [Download Node.js](https://nodejs.org/)
- **npm
- **Angular CLI** (v15+ recommended):
  ```bash
  npm install -g @angular/cli

- npm install -D tailwindcss postcss autoprefixer

- ng serve (To run the frontend)