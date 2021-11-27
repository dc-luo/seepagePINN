for K in 1e-1 5e-2 1e-2 8e-3 6e-3 5e-3 4e-3 2e-3 1e-4; do
    python synthetic_terms.py -K ${K} -n 50000
done;

