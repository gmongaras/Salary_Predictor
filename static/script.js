// This function updates an input field when changed
function update(name) {
	el = document.getElementById(name);
	value = el.value;
	if (value != "") {
		el.classList.add("has-val");
	}
	else {
		el.classList.remove("has-val");
	}
}


// This function is called when the form is submitted to
// validate the form and send errors if needed. It returns
// false if the valid was false and true otherwise.
function validate() {
	valid = true;
	if (validate_helper('ssc_p') == false) {
		valid = false;
		document.getElementById('ssc_p').scrollIntoView();
	}
	if (validate_helper('hsc_p') == false && valid != false) {
		valid = false;
		document.getElementById('hsc_p').scrollIntoView();
	}
	if (validate_helper('degree_p') == false && valid != false) {
		valid = false;
		document.getElementById('degree_p').scrollIntoView();
	}
	if (validate_helper('etest_p') == false && valid != false) {
		valid = false;
		document.getElementById('etest_p').scrollIntoView();
	}
	if (validate_helper('mba_p') == false && valid != false) {
		valid = false;
		document.getElementById('mba_p').scrollIntoView();
	}
	return valid;
}


// This function adds a class to a form input element
// if the input was not valid it returns false if
// the input was valid and true otherwise
function validate_helper(name) {
	el = document.getElementById(name);
	value = el.value;
	if ((value <= 0 || value > 100) && value != "") {
		el.parentElement.classList.add("alert-validate");
		return false;
	}
	else {
		el.parentElement.classList.remove("alert-validate");
		return true;
	}
}