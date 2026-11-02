pipeline {
    agent any

    environment {
        IMAGE_NAME   = "travel-agent"
        IMAGE_TAG    = "${env.BUILD_NUMBER}"
        COMPOSE_FILE = "docker-compose.yml"
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {

        // ── 1. Checkout ───────────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        // ── 2. Lint ───────────────────────────────────────────────────────────
        stage('Lint') {
            steps {
                sh '''
                    pip install --quiet flake8
                    flake8 backend/ ui/ \
                        --select=E,F \
                        --max-line-length=120 \
                        --exclude=__pycache__,.git,backend/data
                '''
            }
        }

        // ── 3. Unit tests ─────────────────────────────────────────────────────
        // Drop this stage if you have no tests/; it will not fail on empty dirs
        stage('Unit Tests') {
            steps {
                sh '''
                    pip install --quiet pytest pytest-asyncio httpx
                    if [ -d "tests" ]; then
                        pytest tests/ -v --tb=short \
                            --junitxml=test-results/results.xml || true
                    else
                        echo "No tests/ directory found — skipping."
                    fi
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results/*.xml'
                }
            }
        }

        // ── 4. Build Docker image ─────────────────────────────────────────────
        stage('Build') {
            steps {
                sh """
                    docker build \
                        -t ${IMAGE_NAME}:${IMAGE_TAG} \
                        -t ${IMAGE_NAME}:latest \
                        .
                """
            }
        }

        // ── 5. Bring services up ──────────────────────────────────────────────
        stage('Start Services') {
            steps {
                // Inject GROQ_API_KEY from Jenkins credentials store
                withCredentials([string(credentialsId: 'GROQ_API_KEY', variable: 'GROQ_API_KEY')]) {
                    sh """
                        echo "GROQ_API_KEY=${GROQ_API_KEY}" > .env
                        echo "LLM_MODEL=llama-3.3-70b-versatile"  >> .env
                        docker-compose -f ${COMPOSE_FILE} up -d
                    """
                }
            }
        }

        // ── 6. Integration smoke test ─────────────────────────────────────────
        stage('Integration Test') {
            steps {
                sh '''
                    echo "Waiting for FastAPI backend..."
                    for i in $(seq 1 15); do
                        curl -sf http://localhost:8000/docs && break
                        echo "  attempt $i/15 — waiting 10s"
                        sleep 10
                    done
                    curl -sf http://localhost:8000/docs \
                        || (echo "Backend did not start in time" && exit 1)

                    echo "Running /plan smoke test..."
                    curl -sf -X POST http://localhost:8000/plan \
                        -H "Content-Type: application/json" \
                        -d '"destination":"Tokyo","days":2,"budget":"Moderate","interests":["Food"],"travel_dates":"December 2026","restrictions":""' \
                        | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'itinerary' in d, 'Missing itinerary key in response'
assert len(d['itinerary']) > 100, 'Itinerary suspiciously short'
print('Smoke test passed. Itinerary length:', len(d['itinerary']))
"
                '''
            }
        }

        // ── 7. MLflow health check ────────────────────────────────────────────
        stage('MLflow Health') {
            steps {
                sh '''
                    curl -sf http://localhost:5000/health \
                        || (echo "MLflow server not healthy" && exit 1)
                    echo "MLflow OK"

                    # Verify at least one experiment was created by the smoke test
                    curl -sf "http://localhost:5000/api/2.0/mlflow/experiments/search" \
                        | python3 -c "
import sys, json
d = json.load(sys.stdin)
exps = d.get('experiments', [])
names = [e['name'] for e in exps]
print('Experiments found:', names)
assert any('travel' in n for n in names), 'travel-agent experiment not found in MLflow'
print('MLflow experiment check passed.')
"
                '''
            }
        }
    }

    // ── Post-build cleanup ────────────────────────────────────────────────────
    post {
        always {
            sh """
                docker-compose -f ${COMPOSE_FILE} down \
                    --volumes --remove-orphans || true
                rm -f .env
            """
            cleanWs()
        }
        success {
            echo "✅ Build ${IMAGE_TAG} passed — ${IMAGE_NAME}:${IMAGE_TAG} is ready."
        }
        failure {
            echo "❌ Build ${IMAGE_TAG} failed — check stage logs above."
        }
    }
}