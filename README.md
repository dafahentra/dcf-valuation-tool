# DCF Valuation Tool

An institutional-grade Discounted Cash Flow (DCF) valuation engine built with Python and Streamlit. This tool leverages Monte Carlo simulations and automated market data fetching to provide comprehensive, probabilistic equity valuations.

Designed for analysts, portfolio managers, and financial researchers who need a robust, transparent, and customizable valuation framework without the overhead of complex spreadsheet models.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/streamlit-framework-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why Fork This Project?

- **Transparent Methodology**: Every calculation, from WACC to terminal value, is fully accessible and modifiable in Python.
- **Probabilistic Approach**: Moves beyond deterministic models by incorporating Monte Carlo simulations for risk and sensitivity analysis.
- **Global Market Coverage**: Pre-configured support for equities across US, UK, Germany, Japan, Hong Kong, India, and China markets.
- **Extensible Architecture**: Clean separation of calculation engine (`dcf_engine.py`), data fetching (`beta_fetcher.py`), and visualization (`visualization.py`).

## Core Capabilities

- **Automated Data Integration**: Real-time extraction of beta coefficients and market parameters.
- **Advanced Growth Modeling**: Flexible projection parameters supporting fixed rates and scenario-based bounds.
- **Scenario & Sensitivity Analysis**: Built-in bull/base/bear case modeling and parameter sensitivity visualization.
- **Data Export**: Complete valuation results exportable to JSON for downstream processing.

## Quick Start

### Requirements

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dafahentra/dcf-valuation-tool.git
cd dcf-valuation-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
streamlit run main.py
```

Navigate to `http://localhost:8501` in your browser.

## Project Structure

```text
dcf-valuation-tool/
├── main.py              # Application entry point and UI layout
├── dcf_engine.py        # Core valuation mathematics and Monte Carlo engine
├── beta_fetcher.py      # Market data and beta coefficient integration
├── visualization.py     # Plotly-based interactive charting
├── styles.py            # Application theming and CSS
└── requirements.txt     # Dependency specifications
```

## Methodology Overview

1. **Cost of Capital (WACC)**: Dynamically calculated based on user-defined capital structure, real-time beta, and region-specific risk-free rates and market premiums.
2. **Cash Flow Projection**: Forward-looking Free Cash Flow (FCF) estimation based on historical trends and defined growth phases.
3. **Terminal Value**: Gordon Growth Model implementation with configurable caps to ensure economic reality.
4. **Monte Carlo Engine**: Stochastically varies revenue growth, margin expansion, and cost of capital to generate a probability density function of the intrinsic value.

## Contributing

We welcome contributions from quantitative analysts, developers, and finance enthusiasts.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AdvancedFeature`)
3. Commit your Changes (`git commit -m 'Add AdvancedFeature'`)
4. Push to the Branch (`git push origin feature/AdvancedFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Always perform your own due diligence before making investment decisions.