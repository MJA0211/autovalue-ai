# Portfolio screenshot plan

All screenshot vehicles must be project-owned examples and must call the
authentic RF05 API. Never capture local paths, browser developer tools, source
rows, raw history databases, credentials, or model files.

## Captured release-candidate views

| View | File | State |
|---|---|---|
| Main landing and input | [`landing-desktop.png`](../screenshots/landing-desktop.png) | Authenticated model ready |
| Successful valuation and calibrated range | [`valuation-result-desktop.png`](../screenshots/valuation-result-desktop.png) | Authentic RF05 result for the synthetic Camry preset |
| ML engineering overview | [`engineering-desktop.png`](../screenshots/engineering-desktop.png) | Authenticated model and frozen evidence |
| Responsive mobile landing | [`landing-mobile-cdp.png`](../screenshots/landing-mobile-cdp.png) | Chromium device metrics at 390 × 844 |

The result image was captured only after the browser submitted the form to the
running FastAPI service. Its value is neither embedded in React nor copied from
a third-party record.

## Final hosted capture sequence

1. Landing/input at 1440 × 1200.
2. Successful common-sedan valuation with point and 90% range visible.
3. A missing-mileage example with its data-quality warning visible.
4. ML engineering final-metrics panel.
5. Experiment decision table showing accepted, rejected, experimental, and
   reference outcomes.
6. Architecture view showing separate reference, shadow, and research paths.
7. River mileage-shift scenario with “Simulation only” and ADWIN telemetry.
8. Mobile landing/result at approximately 390 × 844.
9. Tablet input/result stack at approximately 820 × 1180.

Use the deployed HTTPS URLs for the final set. Review contrast, focus indicators,
text wrapping, clipping, API readiness, and private-data absence before replacing
the local release-candidate images.
