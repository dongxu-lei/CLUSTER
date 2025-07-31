# master/Snakefile

# Load all four subworkflows as modules
module Galton_Board:
    snakefile: "Galton_Board/Snakefile"
    prefix: "Galton_Board"

module Static_Town:
    snakefile: "Static_Town/Snakefile"
    prefix: "Static_Town"

module Dynamic_Town:
    snakefile: "Dynamic_Town/Snakefile"
    prefix: "Dynamic_Town"

module Realistic_City:
    snakefile: "Realistic_City/Snakefile"
    prefix: "Realistic_City"

# Import rules from each subworkflow
use rule * from Galton_Board as Galton_Board_*
use rule * from Static_Town as Static_Town_*
use rule * from Dynamic_Town as Dynamic_Town_*
use rule * from Realistic_City as Realistic_City_*

rule all:
    input:
        rules.Galton_Board_final.output,
        rules.Static_Town_final.output,
        rules.Dynamic_Town_final.input,
        rules.Realistic_City_final.input
    default_target: True

rule Galton_Board:
    input:
        rules.Galton_Board_final.input

rule Static_Town:
    input:
        rules.Static_Town_final.input

rule Dynamic_Town:
    input:
        rules.Dynamic_Town_final.input

rule Realistic_City:
    input:
        rules.Realistic_City_final.input