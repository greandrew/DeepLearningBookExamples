# DeepLearningBookExamples

Practical deep learning notebooks and utilities, organized by chapter.  
This repository is designed for reproducible execution with Docker and Docker Compose, with optional GPU acceleration.

This is the companion repository for *Deep Learning in Quantitative Finance* (Green, 2026):  
https://www.wiley.com/en-ae/Deep+Learning+in+Quantitative+Finance-p-9781119685241

## Stable Release

The latest stable version is **v1.0.0**.

👉 Checkout the release:
git checkout v1.0.0

👉 Or download from:
https://github.com/greandrew/DeepLearningBookExamples/releases

## Repository layout

Each chapter folder contains notebooks plus a chapter-local container setup:

- `Dockerfile`
- `docker-compose.yml`
- notebooks (`*.ipynb`) and helper scripts (`*.py`)

Examples:

- [Chap10RL](Chap10RL)
- [Chap11DV](Chap11DV)
- [Chap12PDE](Chap12PDE)
- [Chap13MC](Chap13MC)
- [Chap14SR](Chap14SR)
- [Chap15Vol](Chap15Vol)
- [Chap16Cal](Chap16Cal)
- [Chap17XVA](Chap17XVA)
- [Chap18MD](Chap18MD)
- [Chap19DH](Chap19DH)

---

## Prerequisites

### Required

- Git
- Docker Engine
- Docker Compose (plugin preferred)

### Optional (for GPU)

- NVIDIA GPU
- NVIDIA driver
- NVIDIA Container Toolkit

---

## Quick start (recommended: Docker Compose v2)

1. Clone the repository.
2. Move into the chapter you want to run.
3. Build and start Jupyter.

```bash
git clone <your-fork-or-this-repo-url>
cd DeepLearningBookExamples/Chap10RL
docker compose up --build
```

Then open:

- http://localhost:8888

To stop:

```bash
docker compose down
```

---

## Using Docker Compose in this repo

Every chapter has its own [docker-compose.yml](Chap10RL/docker-compose.yml)-style config.

Typical service behavior:

- Builds image from local `Dockerfile`
- Starts Jupyter Lab on port `8888`
- Mounts the chapter directory into the container
- Requests all NVIDIA GPUs (if available)

### Important notes

- Most chapter compose files map `8888:8888`, so run one chapter at a time unless you change host ports.
- Some volume mounts use absolute host paths (for example under `~/Documents/...`). If your local path differs, update the `volumes` entry in that chapter’s compose file.

---

## Docker Compose v1 vs v2 (what changed)

You may see both command styles online:

- **Compose v1 (legacy):** `docker-compose ...`
- **Compose v2 (current):** `docker compose ...`

### Key differences

1. **Command name**
	 - v1: standalone binary `docker-compose`
	 - v2: Docker CLI plugin `docker compose`

2. **Installation model**
	 - v1: installed separately
	 - v2: ships as a plugin with modern Docker installs

3. **Lifecycle/support**
	 - v1 is legacy/end-of-life in most setups
	 - v2 is actively maintained and recommended

4. **Compatibility**
	 - Most `docker-compose.yml` files in this repo work in both.
	 - Prefer v2 unless your environment only has v1.

### Command mapping

- `docker-compose up --build` → `docker compose up --build`
- `docker-compose down` → `docker compose down`
- `docker-compose logs -f` → `docker compose logs -f`

---

## Common workflows

### Run a different chapter

```bash
cd Chap11DV
docker compose up --build
```

### Run in background

```bash
docker compose up -d --build
docker compose logs -f
docker compose down
```

### Rebuild after dependency changes

```bash
docker compose build --no-cache
docker compose up
```

---

## Troubleshooting

- **`docker: 'compose' is not a docker command`**  
	Install/enable Docker Compose v2 plugin, or use `docker-compose` if your system only has v1.

- **Port 8888 already in use**  
	Stop other Jupyter/compose services, or edit the chapter `ports` mapping (e.g. `8889:8888`).

- **GPU not visible in container**  
	Verify NVIDIA drivers + container toolkit installation, then restart Docker.

- **Notebook files not visible**  
	Check `volumes` paths in that chapter’s compose file and adjust to your local filesystem.

---

## License

See [LICENSE](LICENSE).
