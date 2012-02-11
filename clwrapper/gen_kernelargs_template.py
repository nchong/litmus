#!/usr/bin/env python
import sys

def generate(n):
  print "template<",
  print ", ".join([ "typename T%d" % i for i in range(n) ]),
  print ">"
  print "inline void kernel_arg(cl_kernel k,",
  print ", ".join([ "T%d &a%d" % (i,i) for i in range(n) ]),
  print ") {"
  print "\n".join([ "  set_kernel_arg(k, %d, a%d);" % (i, i) for i in range(n) ])
  print "}"
  print

def usage():
  print "%s <n>" % sys.argv[0]
  print "  where n is the maximum number of kernel args you want me to generate."

if __name__ == '__main__':
  if len(sys.argv) < 2:
    usage()
    exit(1)
  n = int(sys.argv[1])
  [ generate(n) for n in range(1,n+2) ]
