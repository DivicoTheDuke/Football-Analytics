{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# End-to-end football analytics walkthrough\n",
        "This notebook demonstrates data quality, xG, xT and match analysis using synthetic data."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from football_analytics.data import load_events\n",
        "from football_analytics.quality import validate_events\n",
        "from football_analytics.xg import train_xg, predict_xg\n",
        "from football_analytics.xt import train_xt, apply_xt\n",
        "events = load_events('../data/demo/events.csv')\n",
        "validate_events(events).to_dict()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "xg_artifact = train_xg(events)\n",
        "shots = predict_xg(xg_artifact.model, events)\n",
        "xg_artifact.metrics"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "xt_model = train_xt(events)\n",
        "enriched = apply_xt(xt_model, events)\n",
        "enriched.sort_values('xt_added', ascending=False).head()"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.11"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}