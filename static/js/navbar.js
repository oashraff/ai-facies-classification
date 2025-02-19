let prevScrollPos = window.pageYOffset;
const navbar = document.querySelector('.navbar');
const threshold = 100;

window.addEventListener('mousemove', (e) => {
    if (e.clientY <= threshold) {
        navbar.classList.add('visible');
    } else {
        navbar.classList.remove('visible');
    }
});

window.addEventListener('scroll', () => {
    const currentScrollPos = window.pageYOffset;
    
    if (currentScrollPos < prevScrollPos) {
        navbar.classList.add('visible');
    } else {
        navbar.classList.remove('visible');
    }
    
    prevScrollPos = currentScrollPos;
});

const hamburger = document.querySelector('.hamburger');
const navLinks = document.querySelector('.nav-links');
const spans = document.querySelectorAll('.hamburger span');

hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('active');
    
    // Animate hamburger
    spans[0].classList.toggle('rotate-45');
    spans[1].classList.toggle('opacity-0');
    spans[2].classList.toggle('rotate-negative-45');
});

// Close menu when clicking outside
document.addEventListener('click', (e) => {
    if (!hamburger.contains(e.target) && !navLinks.contains(e.target)) {
        navLinks.classList.remove('active');
        spans.forEach(span => {
            span.classList.remove('rotate-45', 'opacity-0', 'rotate-negative-45');
        });
    }
}); 