<p align="center">
  <img src="fig/logo.png" alt="DINGO Logo" width="300"/>
</p>

# DINGO

**DINGO** — *Dispersion-based INference for Galaxy Observations* — is a Python package designed to fit galaxy kinematics using **both R and C grism spectroscopy**.

It provides robust modeling of galaxy rotation curves, dynamical mass profiles, and kinematic fields through a joint-inference framework leveraging dual-grism data (R+C) from instruments like JWST/NIRCam.

---

## Features (under development)

- 🔵 Joint modeling of R and C grism images.
- 🔵 Arctangent disk rotation curve fitting.
- 🔵 Velocity field generation.
- 🔵 Dynamical mass profile estimation.
- 🔵 Visualization tools for grism images, velocity fields, and residuals.

---

## Installation

For active development:

```bash
git clone git@github.com:Chisen-Lupus/grism-kinematics.git
cd grism-kinematics
pip install -e .
