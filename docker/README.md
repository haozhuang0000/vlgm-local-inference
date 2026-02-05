# How to run

## Single model containers

```bash
# Q-Align (video quality assessment)
docker compose -f docker/docker-compose.qalign.yml up -d --build

# TurboDiffusion (video generation)
docker compose -f docker/docker-compose.turbodiffusion.yml up -d --build

# FastVideo (accelerated video generation)
docker compose -f docker/docker-compose.fastvideo.yml up -d --build
```

## All models (combined container)

```bash
docker compose up -d
```