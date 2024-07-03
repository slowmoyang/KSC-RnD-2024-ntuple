#!/usr/bin/env bash

function assert_var_defined() {
    TARGET=${1}

    if [ -z "${!TARGET}" ]; then
        >&2 echo "${TARGET} not defined: ${!TARGET}"
        exit 1
    else
        echo "${TARGET}: ${!TARGET}"
    fi
}

function assert_file_exists() {
    TARGET=${1}

    if [ ! -f ${!TARGET} ]; then
        >&2 echo "${TARGET} not found: ${!TARGET}"
        exit 1
    else
        echo "${TARGET}: ${!TARGET}"
    fi
}

function assert_file_not_exists() {
    TARGET=${1}

    if [ -f ${!TARGET} ]; then
        >&2 echo "${TARGET} already exists: ${!TARGET}"
        exit 1
    else
        echo "${TARGET}: ${!TARGET}"
    fi
}

function assert_dir_exists() {
    TARGET=${1}

    if [ ! -d ${!TARGET} ]; then
        >&2 echo "${TARGET} not found: ${!TARGET}"
        exit 1
    else
        echo "${TARGET}: ${!TARGET}"
    fi

}


function get_counter() {
    INDEX=${1}
    printf -v COUNTER "%06d" $(( $((${INDEX} - 1 )) / 1000))
    echo ${COUNTER}
}
