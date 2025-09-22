// 导航栏交互
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');

    // 移动端菜单切换
    hamburger.addEventListener('click', function() {
        hamburger.classList.toggle('active');
        navMenu.classList.toggle('active');
    });

    // 点击导航链接时关闭移动端菜单
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            hamburger.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });

    // 滚动时导航栏效果
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.15)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
        }
    });

    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // 技能条动画
    const skillBars = document.querySelectorAll('.skill-progress');
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px 0px -50px 0px'
    };

    const skillObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const skillBar = entry.target;
                const width = skillBar.style.width;
                skillBar.style.width = '0%';
                setTimeout(() => {
                    skillBar.style.width = width;
                }, 200);
            }
        });
    }, observerOptions);

    skillBars.forEach(bar => {
        skillObserver.observe(bar);
    });

    // 卡片悬停效果
    const cards = document.querySelectorAll('.info-card, .research-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // 打字机效果（可选）
    function typeWriter(element, text, speed = 100) {
        let i = 0;
        element.innerHTML = '';
        
        function type() {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        
        type();
    }

    // 页面加载动画
    window.addEventListener('load', function() {
        const heroTitle = document.querySelector('.hero-title');
        if (heroTitle) {
            const originalText = heroTitle.textContent;
            typeWriter(heroTitle, originalText, 50);
        }
    });

    // 返回顶部按钮
    const backToTopButton = document.createElement('button');
    backToTopButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
    backToTopButton.className = 'back-to-top';
    backToTopButton.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        font-size: 18px;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    `;
    
    document.body.appendChild(backToTopButton);

    // 显示/隐藏返回顶部按钮
    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            backToTopButton.style.opacity = '1';
            backToTopButton.style.visibility = 'visible';
        } else {
            backToTopButton.style.opacity = '0';
            backToTopButton.style.visibility = 'hidden';
        }
    });

    // 返回顶部功能
    backToTopButton.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // 悬停效果
    backToTopButton.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.1)';
        this.style.background = '#1d4ed8';
    });

    backToTopButton.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1)';
        this.style.background = '#2563eb';
    });

    // 页面切换动画
    const pageLinks = document.querySelectorAll('a[href$=".html"]');
    pageLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const href = this.getAttribute('href');
            
            // 添加页面切换动画
            document.body.style.opacity = '0';
            document.body.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                window.location.href = href;
            }, 300);
        });
    });

    // 页面加载时恢复透明度
    window.addEventListener('load', function() {
        document.body.style.opacity = '1';
        document.body.style.transform = 'translateY(0)';
    });

    // 懒加载图片（如果有的话）
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });

    images.forEach(img => {
        imageObserver.observe(img);
    });

    // 主题切换功能（可选）
    const themeToggle = document.createElement('button');
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    themeToggle.className = 'theme-toggle';
    themeToggle.style.cssText = `
        position: fixed;
        top: 100px;
        right: 30px;
        width: 50px;
        height: 50px;
        background: #374151;
        color: white;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        font-size: 18px;
        transition: all 0.3s ease;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    `;
    
    document.body.appendChild(themeToggle);

    let isDarkMode = false;
    themeToggle.addEventListener('click', function() {
        isDarkMode = !isDarkMode;
        
        if (isDarkMode) {
            document.body.classList.add('dark-mode');
            this.innerHTML = '<i class="fas fa-sun"></i>';
            this.style.background = '#fbbf24';
            this.style.color = '#1f2937';
        } else {
            document.body.classList.remove('dark-mode');
            this.innerHTML = '<i class="fas fa-moon"></i>';
            this.style.background = '#374151';
            this.style.color = 'white';
        }
    });

    // 表单验证（联系页面使用）
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const inputs = form.querySelectorAll('input, textarea');
            let isValid = true;
            
            inputs.forEach(input => {
                if (input.hasAttribute('required') && !input.value.trim()) {
                    isValid = false;
                    input.style.borderColor = '#ef4444';
                    input.style.boxShadow = '0 0 0 3px rgba(239, 68, 68, 0.1)';
                } else {
                    input.style.borderColor = '#d1d5db';
                    input.style.boxShadow = 'none';
                }
            });
            
            if (isValid) {
                // 这里可以添加实际的表单提交逻辑
                alert('感谢您的留言！我会尽快回复您。');
                form.reset();
            } else {
                alert('请填写所有必填字段。');
            }
        });
    });

    // 搜索功能（博客页面使用）
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const blogPosts = document.querySelectorAll('.blog-post');
            
            blogPosts.forEach(post => {
                const title = post.querySelector('.post-title').textContent.toLowerCase();
                const content = post.querySelector('.post-excerpt').textContent.toLowerCase();
                
                if (title.includes(searchTerm) || content.includes(searchTerm)) {
                    post.style.display = 'block';
                } else {
                    post.style.display = 'none';
                }
            });
        });
    }

    // 博客分类筛选功能
    const filterBtns = document.querySelectorAll('.filter-btn');
    const blogPosts = document.querySelectorAll('.blog-post');
    
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // 移除所有按钮的active类
            filterBtns.forEach(b => b.classList.remove('active'));
            // 添加当前按钮的active类
            this.classList.add('active');
            
            const category = this.getAttribute('data-category');
            
            blogPosts.forEach(post => {
                if (category === 'all' || post.getAttribute('data-category') === category) {
                    post.style.display = 'block';
                } else {
                    post.style.display = 'none';
                }
            });
        });
    });

    // FAQ功能
    const faqItems = document.querySelectorAll('.faq-item');
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        question.addEventListener('click', function() {
            // 关闭其他FAQ项
            faqItems.forEach(otherItem => {
                if (otherItem !== item) {
                    otherItem.classList.remove('active');
                }
            });
            
            // 切换当前FAQ项
            item.classList.toggle('active');
        });
    });

    // 联系表单提交处理
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const submitBtn = this.querySelector('.submit-btn');
            const btnText = submitBtn.querySelector('.btn-text');
            const btnLoading = submitBtn.querySelector('.btn-loading');
            
            // 显示加载状态
            btnText.style.display = 'none';
            btnLoading.style.display = 'flex';
            submitBtn.disabled = true;
            
            // 模拟表单提交
            setTimeout(() => {
                alert('感谢您的留言！我会尽快回复您。');
                this.reset();
                
                // 恢复按钮状态
                btnText.style.display = 'flex';
                btnLoading.style.display = 'none';
                submitBtn.disabled = false;
            }, 2000);
        });
    }
});

// 工具函数
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// 性能优化：使用节流处理滚动事件
const throttledScrollHandler = throttle(function() {
    // 滚动相关的处理逻辑
}, 16); // 约60fps

window.addEventListener('scroll', throttledScrollHandler);
