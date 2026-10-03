---
title: GeoParcel AI
emoji: 🗺️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Building footprints to parcel candidates from drone/satellite
---

# GeoParcel AI — SIH 2026, PS 26012

Upload a drone orthomosaic tile or a satellite-map screenshot, or pick one of the examples. The app segments building footprints with a U-Net (ResNet-34 encoder), regularises them into right-angled polygons, and lets you download them as GeoJSON. A georeferenced GeoTIFF is exported in its own coordinate system.

Outputs are **preliminary parcel candidates for field verification**. They are decision support, not a legal land record.

This Space runs `app.py`, which fetches the code from GitHub at a pinned commit and the model weights from the GitHub release.
Code, training pipeline and evaluation: https://github.com/OMMEOW/SIH26_PS26012
