const form = document.querySelector("#form");

form.addEventListener('submit', e => {
    e.preventDefault();
    fetch('http://localhost:5000')
    .then(res => {
        return res.json()
    })
    .then(obj => {
        console.log(obj)
    })

    for (var i = 0; i < 10000; i++) {
        window.localStorage.setItem('dictionary'+i, 'destination')
    }
});