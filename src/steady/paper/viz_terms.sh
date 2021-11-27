# python experimental_all.py -n 50000 -c 1mm
# python experimental_all.py -n 50000 -c 2mm

# python synthetic_terms.py -K 1e-1 -n 50000
# python synthetic_terms.py -K 1e-2 -n 50000
# python synthetic_terms.py -K 1e-3 -n 50000
# python synthetic_terms.py -K 1e-4 -n 50000

# for i in {0..10}; do
    # python experimental_terms.py -c 2mm -i $i
# done;

for i in {0..11}; do
    python viz_exp_terms.py -n 1mm -i $i -u -t
done;
