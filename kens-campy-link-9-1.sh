#!/bin/bash

export PATH="/usr/local/cuda-9.1/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda-9.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

nvccargs=""

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"
    shift

    case $key in
	-c) # specifies compile only
	    nvccargs="$nvccargs -c "
	    ;;
	-x) # specifies the language
	    nvccargs="$nvccargs -x cu "
	    shift
	    ;;
	-I)
	    nvccargs="$nvccargs -I'$1' "
	    shift
	    ;;
        -W*)
#	    nvccargs="$nvccargs --compiler-options '$key' "
	    ;;
	-o)
            file=$1
	    nvccargs="$nvccargs -o $1 "
	    shift
	    ;;
	-g1)
#	    nvccargs="$nvccargs --compiler-options -g1 "
	    ;;
	-O3)
#	    nvccargs="$nvccargs --compiler-options -O3 "
	    ;;
	-D*)
	    nvccargs="$nvccargs $key "
	    ;;
	-f*)
	    nvccargs="$nvccargs --compiler-options $key "
	    ;;
	-std*)
#	    nvccargs="$nvccargs --compiler-options $key "
	    ;;
	-g*)
	    ;;
	*)
	    nvccargs="$nvccargs $key "
	    ;;
	
    esac
done

nvcc  -dlink $nvccargs
exit 0
