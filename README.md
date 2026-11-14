# Shoulder girdle estimation with an upper-limb exoskeleton

Interactive companion pages to an ICORR paper (in review) on estimating shoulder girdle elevation with an upper-limb exoskeleton.

- `public/index.html`: landing page
- `public/site/`: observability analysis (singular values of the GH-centre Jacobian, least observable girdle motion)
- `public/site_robot/`: estimation from the cuff poses of a generic exoskeleton, with and without thorax and scapula IMUs

Simulation only. The exoskeleton is generic, not a commercial or research device. The model, bone meshes and motions are from Seth A, Matias R, Veloso AP, Delp SL (2016), *A biomechanical model of the scapulothoracic joint to accurately capture scapular kinematics during shoulder movements*, PLoS ONE 11(1): e0141028, [simtk.org/projects/thoracoscapular](https://simtk.org/projects/thoracoscapular).

The pages are static (HTML, JSON, three.js from jsDelivr) and are deployed with GitHub Pages (`.github/workflows/pages.yml`). To view locally:

```
cd public && python -m http.server
```
