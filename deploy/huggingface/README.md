---
title: GeoParcel AI
emoji: 🗺️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Building footprints → parcel candidates from drone/satellite
---

# GeoParcel AI — SIH 2026, PS 26012

Upload a drone orthomosaic tile or a satellite-map screenshot, or pick one of the sample images. The app segments building footprints with a U-Net (ResNet-34 encoder), regularises them into right-angled polygons, and lets you download them as GeoJSON. A georeferenced GeoTIFF is exported in its own coordinate system.

Outputs are **preliminary parcel candidates for field verification**. They are decision support, not a legal land record.

Code, training pipeline and evaluation: https://github.com/OMMEOW/SIH26_PS26012
