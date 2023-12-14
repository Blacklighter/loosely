cur_dir=$(dirname $0)
cd $cur_dir
sh build.sh
pip install --no-index dist/loosely-1.0-py3-none-any.whl --force-reinstall
sleep 2