
function myFunction() {
    var x = document.querySelector('input[name="one"]:checked').value;
    var y = document.querySelector('input[name="two"]:checked').value;
    var a = document.querySelector('input[name="three"]:checked').value;
    var b = document.querySelector('input[name="four"]:checked').value;
    var c = document.querySelector('input[name="five"]:checked').value;
    var z = parseInt(x) + parseInt(y) + parseInt(a) + parseInt(b) + parseInt(c);
    document.getElementById("demo").innerHTML = z
}

