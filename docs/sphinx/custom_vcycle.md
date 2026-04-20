# Custom V-cycle

`vcycle_uniform` est suffisant pour la plupart des cas. Quand tu veux
expérimenter (autre smoother, restriction Galerkin, schedule W-cycle ou
F-cycle, debug d'une étape précise), poisson_cpp expose les briques
individuelles. Cette page assemble un V-cycle récursif depuis zéro
en utilisant `gs_smooth`, `laplacian_fv`, `restrict_avg` et
`prolongate_bilinear`.

## Principe

Sur une grille uniforme cell-centered avec `V = 0` au bord, le V-cycle
récursif fait :

1. *Pré-lissage* : quelques sweeps GS qui tuent l'erreur haute fréquence.
2. *Résidu* : `r = rho - A V` calculé par `laplacian_fv`.
3. *Restriction* : `r_c = restrict_avg(r)` ramène le résidu sur la grille
   2× plus grossière.
4. *Récursion* : on résout `A_c δ_c = r_c` sur la grille grossière, soit
   par lissage exhaustif (au plus grossier), soit par un V-cycle plus
   profond.
5. *Prolongation* : `δ = prolongate_bilinear(δ_c)` ramène le correctif
   sur la grille fine.
6. *Correction + post-lissage* : `V += δ`, puis quelques sweeps GS pour
   nettoyer le bruit haute fréquence introduit par la prolongation.

## Implémentation

```python
import math
import numpy as np
import poisson_cpp as pc


def my_vcycle(V, rho, h, n_pre=2, n_post=2, n_min=4, n_coarse=20):
    """V-cycle récursif maison, équivalent à pc.vcycle_uniform."""
    V = np.asfortranarray(V)
    rho_f = np.asfortranarray(rho)

    # 1. Pré-lissage
    pc.gs_smooth(V, rho_f, h, n_pre)

    # Cas de base : on est sur la grille la plus grossière
    if V.shape[0] <= n_min:
        pc.gs_smooth(V, rho_f, h, n_coarse)
        return V

    # 2. Résidu sur la grille fine
    r = rho - pc.laplacian_fv(V, h)

    # 3. Restriction vers la grille 2× plus grossière
    r_c = pc.restrict_avg(r)
    delta_c = np.zeros_like(r_c)

    # 4. Récursion : on résout A * delta_c = r_c
    delta_c = my_vcycle(delta_c, r_c, 2 * h,
                         n_pre=n_pre, n_post=n_post,
                         n_min=n_min, n_coarse=n_coarse)

    # 5. Prolongation + correction
    V += pc.prolongate_bilinear(delta_c)
    V = np.asfortranarray(V)

    # 6. Post-lissage
    pc.gs_smooth(V, rho_f, h, n_post)
    return V


# Validation : source manufacturée V_exact = sin(pi x) sin(pi y).
N = 64
h = 1.0 / N
xc = (np.arange(N) + 0.5) * h
X, Y = np.meshgrid(xc, xc, indexing="ij")
V_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
rho     = 2 * (math.pi ** 2) * V_exact

V = np.zeros((N, N))
for cycle in range(15):
    V = my_vcycle(V, rho, h)
    r = float(np.max(np.abs(rho - pc.laplacian_fv(V, h))))
    print(f"cycle {cycle:2d} | résidu = {r:.2e}")

print(f"erreur finale vs exacte : {np.max(np.abs(V - V_exact)):.2e}")
```

Le résidu chute d'un facteur ~10 à chaque cycle (taux typique d'un
V-cycle GS bien équilibré). Le résultat final colle à la solution
`pc.vcycle_uniform` à la précision machine.

## Ajouter une variante

Pour passer à un W-cycle, double la récursion à l'étape 4 :

```python
delta_c = my_vcycle(delta_c, r_c, 2 * h, ...)
delta_c = my_vcycle(delta_c, r_c, 2 * h, ...)
```

Pour un smoother plus agressif, augmenter `n_pre`/`n_post` ou intercaler
plusieurs `gs_smooth` avec des paramètres différents.

Pour une restriction full-weighted (moyenne pondérée 9-points au lieu de
4-points), il faut l'écrire en numpy : `restrict_avg` est l'équivalent
4-points.
