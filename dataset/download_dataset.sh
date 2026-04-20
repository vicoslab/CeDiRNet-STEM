#!/bin/bash

CURDIR=$(dirname "$(realpath "$0")")

wget https://data.vicos.si/skokec/STEM/nanoparticles.zip -O ${CURDIR}/nanoparticles.zip && unzip ${CURDIR}/nanoparticles.zip -d ${CURDIR}

