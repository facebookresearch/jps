#include <iostream>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/interpreter.h>

int main (const int argc, char ** argv)
{
const char * argvv [] = {"" /* name of program, not relevant */, "--silent"};

  octave_main (2, (char **) argvv, true /* embedded */);

  octave_value_list functionArguments;

  functionArguments (0) = 2;
  functionArguments (1) = "D. Humble";

  Matrix inMatrix (2, 3);

  inMatrix (0, 0) = 10;
  inMatrix (0, 1) = 9;
  inMatrix (0, 2) = 8;
  inMatrix (1, 0) = 7;
  inMatrix (1, 1) = 6;
  functionArguments (2) = inMatrix;

  std::cout << "about to exec" << std::endl;
const octave_value_list result = feval ("exampleOctaveFunction", functionArguments, 1);

  std::cout << "resultScalar is " << result (0).scalar_value () << std::endl;
  std::cout << "resultString is " << result (1).string_value () << std::endl;
  std::cout << "resultMatrix is\n" << result (2).matrix_value ();

  //do_octave_atexit ();*/
}

/*
int
main (void)
{
  string_vector argv (2);
  argv(0) = "embedded";
  argv(1) = "-q";

  octave_main (2, argv.c_str_vec (), 1);

  octave_idx_type n = 2;
  octave_value_list in;

  for (octave_idx_type i = 0; i < n; i++)
    in(i) = octave_value (5 * (i + 2));

  std::cout << "about to exec\n";

  octave_value_list out = feval ("test", in , 1);

  if (! error_state && out.length () > 0)
    std::cout << "GCD of ["
              << in(0).int_value ()
              << ", "
              << in(1).int_value ()
              << "] is " << out(0).int_value ()
              << std::endl;
  else
    std::cout << "invalid\n";

  //clean_up_and_exit (0);
}*/
