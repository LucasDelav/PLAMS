"""
Module de visualisation.

Ce module fournit des outils pour générer des graphiques de corrélation
entre les propriétés moléculaires calculées par les différents pipelines.
"""

# Import batch correlations (inter-molecule comparisons)
from .batch_correlations import (
    generate_batch_plots,
    plot_correlation_heatmap_batch,
    plot_property_vs_potential,
    generate_all_correlations,
    create_correlation_grid,
    generate_frontier_orbital_correlations,
)

# Import frequency correlations
from .frequency_correlations import (
    plot_top_correlations,
    plot_top_focused_correlations,
    plot_focused_correlation,
    plot_individual_correlations,
    plot_focused_individual_correlations,
    generate_all_frequency_plots,
)

# Import density correlations
from .density_correlations import (
    plot_top_density_correlations,
    plot_koopmans_comparison,
    plot_density_individual,
    generate_all_density_plots,
)

__all__ = [
    # Batch correlations
    'generate_batch_plots',
    'plot_correlation_heatmap_batch',
    'plot_property_vs_potential',
    'generate_all_correlations',
    'create_correlation_grid',
    'generate_frontier_orbital_correlations',

    # Frequency correlations
    'plot_top_correlations',
    'plot_top_focused_correlations',
    'plot_focused_correlation',
    'plot_individual_correlations',
    'plot_focused_individual_correlations',
    'generate_all_frequency_plots',

    # Density correlations
    'plot_top_density_correlations',
    'plot_koopmans_comparison',
    'plot_density_individual',
    'generate_all_density_plots',
]
