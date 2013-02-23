#!/bin/sh

gnuplot >projector_vs_mallet_ndocs.eps <<EOF
  set terminal postscript eps size 4,3
  set title "vocab size 1000; avg doc length 100"
  set xlabel "num docs"
  set ylabel "seconds"
  set logscale xy
  set xtics (1000,2000,3000,4000)
  plot [800:5000] [1:200] "-" title "Projector" with linespoints, "-" title "Mallet LDA" with linespoints, x*0.007 title "reference: x proportional to y" with lines
        1000 1.98
        2000 3.14
        3000 3.95
        4000 5.15
    e
        1000 18.38
        2000 35.84
        3000 53.97
        4000 69.7
    e
EOF

epstopdf projector_vs_mallet_ndocs.eps

gnuplot >projector_vs_mallet_doclen.eps <<EOF
  set terminal postscript eps size 4,3
  set title "vocab size 1000; 1000 documents"
  set xlabel "average document length"
  set ylabel "seconds"
  set logscale xy
  set xtics (100,200,300,400)
  plot [80:500] [1:200] "-" title "Projector" with linespoints, "-" title "Mallet LDA" with linespoints, x*0.05 title "reference: x proportional to y" with lines
        100 2.08
        200 2.25
        300 2.42
        400 2.67
    e
        100 18.39
        200 34.86
        300 52.02
        400 69.09
    e
EOF

epstopdf projector_vs_mallet_doclen.eps
