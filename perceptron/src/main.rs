use ndarray::Array;

fn main() {

    let a = Array::from_elem((3,3), 1.);
    let b = Array::from_elem((3,3), 2.);

    a.ncols();

    println!("{:?}", a.dot(&b));
}
