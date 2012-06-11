#!/usr/bin/env python
from __future__ import unicode_literals

import os, sys, re

from pylab import *

# define a list of text tests to run
strings = [
    r'$\mathcircled{123} \mathrm{\mathcircled{123}} \mathbf{\mathcircled{123}}$',
    r'$\mathsf{Sans \Omega} \mathrm{\mathsf{Sans \Omega}} \mathbf{\mathsf{Sans \Omega}}$',
    r'$\mathtt{Monospace}$',
    r'$\mathcal{CALLIGRAPHIC}$',
    r'$\mathbb{Blackboard \pi}$',
    r'$\mathrm{\mathbb{Blackboard \pi}}$',
    r'$\mathbf{\mathbb{Blackboard \pi}}$',
    r'$\mathfrak{Fraktur} \mathbf{\mathfrak{Fraktur}}$',
    r'$\mathscr{Script}$']


def create_plot():
    figure(figsize=(8, (len(strings) * 1) + 2))

    # make a full figure axes
    ax = axes([0, 0, 1, 1])
    # hide the x axis and y axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # set convenient axis data limits. 
    axis([0, 1, len(strings), 0])

    # add the text
    for i, s in enumerate(strings):
        text(0.5, i + 0.5, s, fontsize=32, horizontalalignment='center')

    show()


# provide a command line interface to produce the equivalent latex
if '--latex' not in sys.argv:
    create_plot()
else:
    fd = open("stix_fonts_examples.ltx", "w")
    fd.write("\\documentclass{article}\n")
    fd.write("\\begin{document}\n")
    fd.write("\\begin{enumerate}\n")

    for i, s in enumerate(strings):
        s = re.sub(r"(?<!\\)\$", "$$", s)
        fd.write("\\item %s\n" % s)

    fd.write("\\end{enumerate}\n")
    fd.write("\\end{document}\n")
    fd.close()

    os.system("pdflatex stix_fonts_examples.ltx")