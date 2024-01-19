
CURR_DIR=$(dirname $0)
# echo $CURR_DIR
cd $CURR_DIR
cd ..

echo "==== MTF ===="
python test/test.py 1 0 0 0 
python data-scripts/generate_avg.py $1 mtf

echo "==== GF ====="
python test/test.py 1 0 1 1
python data-scripts/generate_avg.py $1 g_fair

echo "==== RR ====="
python test/test.py 1 0 1 2
python data-scripts/generate_avg.py $1 rr

echo "==== FTF ===="
python test/test.py 1 0 2 3
python data-scripts/generate_avg.py $1 themis

python data-scripts/generate_plots.py -n 20 -t "Power of Two Choices"