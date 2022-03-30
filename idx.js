const form = document.querySelector("#form");

form.addEventListener('submit', e => {
    e.preventDefault();
    fetch('http://localhost:5000', {
        method: 'POST',
    })
    .then(res => {
        return res.json()
    })
    .then(obj => {
        console.log(obj)
    })

    return false;
});