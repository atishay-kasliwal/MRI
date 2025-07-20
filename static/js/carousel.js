let currentSlide = 0;
let totalSlides = 6;

function initializeCarousel() {
    const carousel = document.getElementById('carousel');
    totalSlides = parseInt(carousel.getAttribute('data-slide-count')) || 4;
    showSlide(0);
}

function showSlide(n) {
    currentSlide = (n + totalSlides) % totalSlides;
    const carousel = document.getElementById('carousel');
    carousel.style.transform = `translateX(-${currentSlide * 100}%)`;
    
    // Update indicators
    const indicators = document.querySelectorAll('.indicator');
    indicators.forEach((indicator, index) => {
        indicator.classList.toggle('active', index === currentSlide);
    });
}

function changeSlide(direction) {
    showSlide(currentSlide + direction);
}

function goToSlide(n) {
    showSlide(n);
}

// Auto-advance slides every 5 seconds
setInterval(() => {
    changeSlide(1);
}, 5000); 