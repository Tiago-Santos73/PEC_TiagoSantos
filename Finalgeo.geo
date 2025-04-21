// Gmsh project created on Mon Apr 21 19:52:40 2025
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {1, 0, 0, 1.0};
//+
Line(1) = {2, 1};
//+
Line(2) = {1, 5};
//+
Line(3) = {5, 3};
//+
Line(4) = {3, 2};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Surface(1) = {1};
//+
Physical Curve("Left", 5) = {1};
//+
Physical Curve("Up", 6) = {4};
//+
Physical Curve("Down", 7) = {2};
//+
Physical Curve("Right", 8) = {3};
//+
Physical Surface("Surface", 9) = {1};
//+
Transfinite Curve {1, 4, 3, 2} = 70 Using Progression 1;
