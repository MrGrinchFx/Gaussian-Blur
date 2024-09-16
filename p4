#!/bin/bash

# Safer execution
# -e: exit immediately if a command fails
# -E: Safer -e option for traps
# -u: fail if a variable is used unset
# -o pipefail: exit immediately if command in a pipe fails
#set -eEuo pipefail
# -x: print each command before executing (great for debugging)
#set -x

# Convenient values
SCRIPT_NAME=$(basename $BASH_SOURCE)

# Default program values
TEST_CASE="all"

#
# Logging helpers
#
log() {
    echo -e "${*}"
}
info() {
    log "Info: ${*}"
}
warning() {
    log "Warning: ${*}"
}
error() {
    log "Error: ${*}"
}
die() {
    error "${*}"
    exit 1
}

#
# Line comparison
#
select_line() {
    # 1: string
    # 2: line to select
    echo "$(echo "${1}" | sed "${2}q;d")"
}

fail() {
    # 1: got
    # 2: expected
    log "Fail: got '${1}' but expected '${2}'"
}

pass() {
    # got
    log "Pass: ${1}"
}

compare_lines() {
    # 1: output
    # 2: expected
    # 3: score (output)
    declare -a output_lines=("${!1}")
    declare -a expect_lines=("${!2}")
    local __score=$3
    local partial="0"

    # Amount of partial credit for each correct output line
    local step=$(bc -l <<< "1.0 / ${#expect_lines[@]}")

    # Compare lines, two by two
    for i in ${!output_lines[*]}; do
        if [[ "${output_lines[${i}]}" == "${expect_lines[${i}]}" ]]; then
            pass "${output_lines[${i}]}"
            partial=$(bc <<< "${partial} + ${step}")
        else
            fail "${output_lines[${i}]}" "${expect_lines[${i}]}" ]]
        fi
    done

    # Return final score
    eval ${__score}="'${partial}'"
}

#
# Run generic test case
#
run_test_case() {
	#1: executable name
	local exec="${1}"

    [[ -x "$(command -v ${exec})" ]] || \
        die "Cannot find executable '${exec}'"

    # These are global variables after the test has run so clear them out now
    unset STDOUT STDERR RET

    # Create temp files for getting stdout and stderr
    local outfile=$(mktemp)
    local errfile=$(mktemp)

    "${@}" >${outfile} 2>${errfile}

    # Get the return status, stdout and stderr of the test case
    RET="${?}"
    STDOUT=$(cat "${outfile}")
    STDERR=$(cat "${errfile}")

    # Clean up temp files
    rm -f "${outfile}"
    rm -f "${errfile}"
}

run_time() {
    #1: num repetitions
    local reps="${1}"
    shift
    local exec="${1}"

    [[ -x ${exec} ]] || \
        die "Cannot find executable '${exec}'"

    # These are global variables after the test has run so clear them out now
    unset PERF

    for i in $(seq ${reps}); do
        # Create temp files for getting stdout and stderr
        local outfile=$(mktemp)
        local errfile=$(mktemp)

        TIME="%e" /usr/bin/time "${@}" >${outfile} 2>${errfile}

        # Last line of stderr
        local t=$(cat "${errfile}" | tail -n1)

        # Check it's the right format
        if [[ ! "${t}" =~ ^[0-9]{1,3}\.[0-9]{2}$ ]]; then
            die "Wrong timing output '${t}'"
        fi

        # Keep the best timing
        if [ -z "${PERF}" ]; then
            PERF=${t}
        elif (( $(bc <<<"${t} < ${PERF}") )); then
            PERF=${t}
        fi

        # Clean up temp files
        rm -f "${outfile}"
        rm -f "${errfile}"
    done
}

#
# Test cases
#
TEST_CASES=()

#
# Correctness
#
gaussian_blur_serial_correct() {
    local arg_im=(city  lenna   gentilhomme ucd_pavilion)
    local arg_sz=(256   512     1024        15mp)
    local arg_sg=(2.3   8.6     4.2         1)

    local line_array=()
    local corr_array=()

    for i in $(seq 0 3); do
        run_test_case ./gaussian_blur_serial \
            "${arg_im[${i}]}_${arg_sz[${i}]}.pgm" \
            "${arg_im[${i}]}_${arg_sz[${i}]}_${arg_sg[${i}]}.pgm" \
            "${arg_sg[${i}]}"
        run_test_case compare -channel gray -fuzz 1% -metric AE \
            "ref_${arg_im[${i}]}_${arg_sz[${i}]}_${arg_sg[${i}]}.pgm" \
            "${arg_im[${i}]}_${arg_sz[${i}]}_${arg_sg[${i}]}.pgm" \
            diff.pgm

        line_array+=("$(select_line "${STDERR}" "1")")
        corr_array+=("0")

        rm -f "${arg_im[${i}]}_${arg_sz[${i}]}_${arg_sg[${i}]}.pgm"
        rm -f diff.pgm
    done

    local score
    compare_lines line_array[@] corr_array[@] score
    log "${score}"
}
TEST_CASES+=("gaussian_blur_serial_correct")

gaussian_blur_cuda_correct() {
    local arg_im=(city  lenna   gentilhomme ucd_pavilion)
    local arg_sz=(256   512     1024        15mp)
    local arg_sg=(2.3   8.6     4.2         1)

    local line_array=()
    local corr_array=()

    for i in $(seq 0 3); do
        run_test_case ./gaussian_blur_cuda \
            "${arg_im[${i}]}_${arg_sz[${i}]}.pgm" \
            "${arg_im[${i}]}_${arg_sz[${i}]}_${arg_sg[${i}]}.pgm" \
            "${arg_sg[${i}]}"
        run_test_case compare -channel gray -fuzz 1% -metric AE \
            "ref_${arg_im[${i}]}_${arg_sz[${i}]}_${arg_sg[${i}]}.pgm" \
            "${arg_im[${i}]}_${arg_sz[${i}]}_${arg_sg[${i}]}.pgm" \
            diff.pgm

        line_array+=("$(select_line "${STDERR}" "1")")
        corr_array+=("0")

        rm -f "${arg_im[${i}]}_${arg_sz[${i}]}_${arg_sg[${i}]}.pgm"
        rm -f diff.pgm
    done

    local score
    compare_lines line_array[@] corr_array[@] score
    log "${score}"
}
TEST_CASES+=("gaussian_blur_cuda_correct")

#
# Speed
#
NREPS=2
gaussian_blur_serial_speed()
{
    run_time ${NREPS} ./ref_gaussian_blur_serial \
        ucd_pavilion_15mp.pgm ucd_pavilion_15mp_2.pgm 2
    local ref_perf=${PERF}

    rm -f "ucd_pavilion_15mp_2.pgm"

    run_time ${NREPS} ./gaussian_blur_serial \
        ucd_pavilion_15mp.pgm ucd_pavilion_15mp_2.pgm 2
    local tst_perf=${PERF}

    rm -f "ucd_pavilion_15mp_2.pgm"

    local ratio=$(bc -l <<<"${tst_perf} / ${ref_perf}")
    log "${ratio}"
}
TEST_CASES+=("gaussian_blur_serial_speed")

gaussian_blur_cuda_speed()
{
    run_time ${NREPS} ./ref_gaussian_blur_cuda \
        ucd_pavilion_15mp.pgm ucd_pavilion_15mp_2.pgm 2
    local ref_perf=${PERF}

    rm -f "ucd_pavilion_15mp_2.pgm"

    run_time ${NREPS} ./gaussian_blur_cuda \
        ucd_pavilion_15mp.pgm ucd_pavilion_15mp_2.pgm 2
    local tst_perf=${PERF}

    rm -f "ucd_pavilion_15mp_2.pgm"

    local ratio=$(bc -l <<<"${tst_perf} / ${ref_perf}")
    log "${ratio}"
}
TEST_CASES+=("gaussian_blur_cuda_speed")

#
# Main functions
#
parse_argvs() {
    local OPTIND opt

    while getopts "h?s:t:" opt; do
        case "$opt" in
            h|\?)
                echo "${SCRIPT_NAME}: [-t <test_case>]" 1>&2
                exit 0
                ;;
            t)  TEST_CASE="${OPTARG}"
                ;;
        esac
    done
}

check_vals() {
    # Check test case
    [[ " ${TEST_CASES[@]} all " =~ " ${TEST_CASE} " ]] || \
        die "Cannot find test case '${TEST_CASE}'"
    }

grade() {
    # Run test case(s)
    if [[ "${TEST_CASE}" == "all" ]]; then
        # Run all test cases
        for t in "${TEST_CASES[@]}"; do
            log "--- Running test case: ${t} ---"
            ${t}
            log "\n"
        done
    else
        # Run specific test case
        ${TEST_CASE}
    fi
}

parse_argvs "$@"
check_vals
grade
