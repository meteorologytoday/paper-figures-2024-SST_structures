#!/bin/bash

# Wong, B. (2011). Color blindness. nature methods, 8(6), 441.
BW8color = {
    'black'         : (  0,   0,   0),
    'orange'        : (230, 159,   0),
    'skyblue'       : ( 86, 180, 233),
    'bluishgreen'   : (  0, 158, 115),
    'yellow'        : (240, 228,  66),
    'blue'          : (  0, 114, 178),
    'vermillion'    : (213,  94,   0),
    'reddishpurple' : (204, 121, 167),
}

for k, c in BW8color.items():
    BW8color[k] = (c[0]/255, c[1]/255, c[2]/255)





