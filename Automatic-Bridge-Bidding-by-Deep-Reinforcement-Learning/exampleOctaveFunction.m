function [resultScalar, resultString, resultMatrix] = exampleOctaveFunction (inScalar, inString, inMatrix)

  resultScalar = (inScalar * pi);
  resultString = strcat ('Good morning Mr. ', inString);
  resultMatrix = (inMatrix + 1);

endfunction
