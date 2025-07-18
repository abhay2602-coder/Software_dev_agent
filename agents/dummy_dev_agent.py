import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from models.task import Task
from models.plan import Plan
from models.enums import TaskStatus
from parse.websocket_manager import WebSocketManager
from utils.llm_setup import ask_llm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output folder for dev agent results
DEV_OUTPUT_DIR = Path("/workspaces/Software_dev_agent/generated_code/dev_outputs")
DEV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DevAgent:
    def cleanup_all_outputs(self):
        """
        Deletes all files and folders in the dev_outputs directory.
        """
        import shutil
        if DEV_OUTPUT_DIR.exists():
            for item in DEV_OUTPUT_DIR.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.warning(f"Failed to delete {item}: {e}")
    def __init__(self, websocket_manager: WebSocketManager = None):
        self.agent_id = "dev_agent"
        self.websocket_manager = websocket_manager or WebSocketManager()
        self.current_plan = None
        self.plan_dir = Path("/workspaces/Software_dev_agent/generated_code/plans/parsed")

    def clear_task_output(self, task_id: str):
        """
        Clears output files and directories for a specific task in the dev_outputs folder.
        """
        for item in DEV_OUTPUT_DIR.iterdir():
            if item.is_dir() and item.name.startswith(f"{task_id}_"):
                for subitem in item.iterdir():
                    try:
                        if subitem.is_file():
                            subitem.unlink()
                        elif subitem.is_dir():
                            import shutil
                            shutil.rmtree(subitem)
                    except Exception as e:
                        logger.warning(f"Failed to delete {subitem}: {e}")
                try:
                    item.rmdir()
                except Exception as e:
                    logger.warning(f"Failed to delete directory {item}: {e}")

    def _get_system_prompt(self) -> str:
        return """You are a **Senior Full-Stack Software Developer AI Agent** with 10+ years of experience in production-grade software development.

## Your Core Expertise:
- **Backend Development**: Python (FastAPI, Django, Flask), Node.js, Java, Go
- **Frontend Development**: React, Vue.js, Angular, HTML/CSS, JavaScript/TypeScript
- **Database Systems**: PostgreSQL, MySQL, MongoDB, Redis, SQLite
- **Cloud & DevOps**: AWS, Docker, Kubernetes, CI/CD pipelines
- **Architecture Patterns**: Microservices, REST APIs, GraphQL, Event-driven architecture
- **Security**: Authentication, authorization, input validation, SQL injection prevention
- **Performance**: Caching, database optimization, async programming, load balancing
- **Testing**: Unit tests, integration tests, mocking, test-driven development

## Code Quality Standards:
✅ **Production-Ready**: Code that can be deployed immediately to production
✅ **Security-First**: Implement proper authentication, input validation, and security headers
✅ **Performance-Optimized**: Efficient algorithms, proper database indexing, caching strategies
✅ **Error Handling**: Comprehensive exception handling and logging
✅ **Documentation**: Clear docstrings, comments, and inline documentation
✅ **Testing**: Include unit tests and integration tests where applicable
✅ **Scalability**: Design for horizontal scaling and high availability
✅ **Maintainability**: Clean, readable code following SOLID principles
✅ **Standards Compliance**: Follow PEP 8, ESLint, and industry best practices

## Implementation Requirements:
1. **Complete Implementation**: Provide full, working code - not pseudocode or snippets
2. **Multiple Files**: If needed, create separate files for models, services, utilities, tests
3. **Configuration**: Include environment variables, config files, and setup instructions
4. **Dependencies**: List all required packages and versions
5. **Database**: Include migration scripts, schema definitions, and seed data
6. **API Documentation**: Provide OpenAPI/Swagger specs for APIs
7. **Deployment**: Include Docker files, deployment scripts, and infrastructure code
8. **Monitoring**: Add logging, metrics, and health checks

## Security Implementation:
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CSRF tokens
- Rate limiting
- Authentication middleware
- Authorization checks
- Secure headers
- Password hashing
- JWT token management

## Performance Optimization:
- Database query optimization
- Caching strategies (Redis, in-memory)
- Async/await patterns
- Connection pooling
- Lazy loading
- Pagination
- Compression
- CDN integration

## Code Structure:
```
project/
├── app/
│   ├── models/
│   ├── services/
│   ├── controllers/
│   ├── middleware/
│   ├── utils/
│   └── tests/
├── config/
├── migrations/
├── docker/
├── docs/
└── scripts/
```

## Response Format:
Provide complete, production-ready code with:
- All necessary imports and dependencies
- Proper error handling and logging
- Security implementations
- Performance optimizations
- Unit tests
- Configuration files
- Setup and deployment instructions

Your code should be immediately deployable to production environments."""

    def _construct_prompt(self, task: Task) -> str:
        return f"""
## Task Details:
**Title**: {task.title}
**Description**: {task.description}
**Estimated Hours**: {task.estimated_hours}
**Complexity**: {task.complexity}
**Dependencies**: {task.dependencies}

## Implementation Requirements:

### 1. **Complete Production Implementation**
- Provide full, working code that can be deployed to production immediately
- Include all necessary files, configurations, and dependencies
- Implement proper error handling, logging, and monitoring
- Add comprehensive security measures and input validation

### 2. **Architecture & Design**
- Follow microservices architecture patterns where applicable
- Implement clean code principles and SOLID design patterns
- Use appropriate design patterns (Factory, Strategy, Observer, etc.)
- Ensure scalability and maintainability

### 3. **Security Implementation**
- Implement authentication and authorization
- Add input validation and sanitization
- Prevent SQL injection, XSS, and CSRF attacks
- Include rate limiting and security headers
- Use secure password hashing and JWT tokens

### 4. **Performance Optimization**
- Implement caching strategies (Redis, in-memory)
- Optimize database queries and use proper indexing
- Add async/await patterns for I/O operations
- Implement connection pooling and resource management
- Add pagination for large datasets

### 5. **Database Integration**
- Create proper database models and relationships
- Include migration scripts and schema definitions
- Add database connection pooling and transaction management
- Implement proper indexing strategies
- Add seed data and test fixtures

### 6. **API Development** (if applicable)
- Create RESTful APIs with proper HTTP status codes
- Implement input validation and serialization
- Add API documentation (OpenAPI/Swagger)
- Include rate limiting and authentication middleware
- Add comprehensive error responses

### 7. **Frontend Implementation** (if applicable)
- Create responsive, accessible UI components
- Implement proper state management
- Add form validation and error handling
- Include loading states and user feedback
- Optimize for performance and SEO

### 8. **Testing Strategy**
- Include unit tests with high coverage
- Add integration tests for APIs and database operations
- Include mocking for external dependencies
- Add performance and security tests
- Include test fixtures and data

### 9. **Configuration & Environment**
- Create environment-specific configuration files
- Use environment variables for sensitive data
- Include Docker configuration and deployment scripts
- Add CI/CD pipeline configuration
- Include monitoring and logging setup

### 10. **Documentation & Deployment**
- Provide comprehensive README with setup instructions
- Include API documentation and code comments
- Add deployment guides and troubleshooting
- Include performance benchmarks and monitoring setup
- Add backup and disaster recovery procedures

## Technology Stack Recommendations:
- **Backend**: FastAPI/Django + PostgreSQL + Redis + Celery
- **Frontend**: React/Vue.js + TypeScript + Tailwind CSS
- **Database**: PostgreSQL with proper indexing and connection pooling
- **Caching**: Redis for session management and API caching
- **Authentication**: JWT tokens with refresh token rotation
- **Deployment**: Docker + Kubernetes + AWS/GCP
- **Monitoring**: Prometheus + Grafana + ELK Stack

## Expected Deliverables:
1. **Complete source code** with all necessary files
2. **Configuration files** (environment, Docker, etc.)
3. **Database migrations** and schema definitions
4. **Unit and integration tests** with good coverage
5. **API documentation** (if applicable)
6. **Deployment scripts** and infrastructure code
7. **Comprehensive README** with setup instructions
8. **Performance benchmarks** and optimization notes

## Code Quality Checklist:
- [ ] All code is production-ready and deployable
- [ ] Comprehensive error handling and logging
- [ ] Security measures implemented (auth, validation, etc.)
- [ ] Performance optimizations applied
- [ ] Unit tests with good coverage
- [ ] API documentation provided
- [ ] Configuration management setup
- [ ] Deployment scripts included
- [ ] Monitoring and health checks added
- [ ] Documentation and setup instructions complete

Please provide a complete, production-ready implementation that follows enterprise-grade software development practices."""

    async def execute_task(self, task: Task) -> Task:
        """
        Executes a single development task.
        Updates the task's status and broadcasts messages via WebSocket.
        Returns the updated Task object.
        """
        logger.info(f"Dev Agent: Starting task '{task.title}' (ID: {task.id})")
        task.status = TaskStatus.IN_PROGRESS # Set status at the start

        await self.websocket_manager.broadcast_message({
            "agent_id": self.agent_id,
            "type": "task_status_update",
            "task_id": task.id,
            "status": task.status.value,
            "message": f"Dev Agent started task: '{task.title}'",
            "timestamp": datetime.now().isoformat()
        })
        
        # Create task-specific directory for outputs
        # Ensure the directory name is safe for file systems
        safe_task_title = "".join(c if c.isalnum() else "_" for c in task.title).lower()[:50]
        task_dir = DEV_OUTPUT_DIR / f"{task.id}_{safe_task_title}"
        
        # If the task directory already exists, clear it before starting
        if task_dir.exists():
            self.clear_task_output(task.id)
            
        # Create the task directory
        task_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            system_prompt = self._get_system_prompt()
            user_prompt = self._construct_prompt(task)
            
            # Ask the LLM with await
            code_output = await ask_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model="gemini-2.5-pro",
                temperature=0.3  # Lower temperature for more consistent, production-ready code
            )
            
            # Save main implementation file
            main_file = task_dir / "implementation.py"
            try:
                main_file.write_text(code_output, encoding="utf-8")
                # Notify frontend about the new implementation file
                await self.websocket_manager.broadcast_message({
                    "type": "file_generated",
                    "file_name": str(main_file.relative_to(DEV_OUTPUT_DIR)),
                    "content": code_output,
                    "file_type": "python"
                })
            except Exception as file_error:
                logger.error(f"DevAgent: Failed to write implementation file for task {task.id}: {file_error}", exc_info=True)
                code_output = f"Error writing file: {file_error}\n\nOriginal LLM Output:\n{code_output}" # Prepend error to content
                main_file.write_text(code_output, encoding="utf-8") # Still attempt to write, with error info
            
            # Save task metadata
            metadata = {
                "task_id": task.id,
                "title": task.title,
                "description": task.description,
                "complexity": task.complexity,
                "estimated_hours": task.estimated_hours,
                "dependencies": task.dependencies,
                "completed_at": datetime.now().isoformat(),
                "output_files": [str(main_file.relative_to(DEV_OUTPUT_DIR))] # Store relative path
            }
            
            metadata_file = task_dir / "task_metadata.json"
            metadata_json = json.dumps(metadata, indent=2)
            metadata_file.write_text(metadata_json, encoding="utf-8")
            
            # Notify frontend about the metadata file
            await self.websocket_manager.broadcast_message({
                "type": "file_generated",
                "file_name": str(metadata_file.relative_to(DEV_OUTPUT_DIR)),
                "content": metadata_json,
                "file_type": "json"
            })
            
            task.status = TaskStatus.COMPLETED # Mark task as completed
            logger.info(f"Dev Agent: Task '{task.title}' (ID: {task.id}) completed successfully.")
            await self.websocket_manager.broadcast_message({
                "agent_id": self.agent_id,
                "type": "task_status_update",
                "task_id": task.id,
                "status": task.status.value,
                "output_directory": str(task_dir.relative_to(DEV_OUTPUT_DIR)), # Relative path for UI
                "main_file": str(main_file.relative_to(DEV_OUTPUT_DIR)), # Relative path for UI
                "message": f"Dev Agent completed task: '{task.title}'",
                "timestamp": datetime.now().isoformat()
            })
            
            return task # Return the updated task object
            
        except Exception as e:
            logger.error(f"DevAgent failed on task {task.id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILED # Mark task as failed
            await self.websocket_manager.broadcast_message({
                "agent_id": self.agent_id,
                "type": "task_status_update",
                "task_id": task.id,
                "status": task.status.value,
                "error": str(e),
                "message": f"Dev Agent failed task: '{task.title}': {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            return task # Return the updated (failed) task object

    # The methods below (_get_latest_plan, load_current_plan, process_plan_tasks, _save_updated_plan)
    # are for a batch processing workflow where the DevAgent fetches and processes
    # a full plan. In a streaming workflow where main.py passes individual tasks,
    # these methods are not the primary entry points for task execution.
    # I'm commenting them out for clarity, assuming you will rely on main.py
    # calling `execute_task` directly per task. If you have another use case
    # for them (e.g., a batch mode or a resume feature), you can uncomment and adapt.

    # def _get_latest_plan(self) -> Optional[Path]:
    #     """Find the most recent plan file in the parsed directory."""
    #     if not self.plan_dir.exists():
    #         return None
    #     plan_files = list(self.plan_dir.glob("plan_*.json"))
    #     if not plan_files:
    #         return None
    #     return max(plan_files, key=lambda p: p.stat().st_mtime)

    # def load_current_plan(self) -> Optional[Plan]:
    #     """Load the most recent plan from the parsed plans directory."""
    #     plan_file = self._get_latest_plan()
    #     if not plan_file:
    #         logger.warning("No plan files found")
    #         return None
    #     
    #     try:
    #         with open(plan_file, 'r') as f:
    #             plan_data = json.load(f)
    #         self.current_plan = Plan(**plan_data)
    #         return self.current_plan
    #     except Exception as e:
    #         logger.error(f"Failed to load plan: {e}")
    #         return None

    # async def process_plan_tasks(self, plan: Optional[Plan] = None):
    #     """
    #     Process all dev_agent tasks from the given plan.
    #     This method is for a batch processing scenario and might not be used
    #     if main.py orchestrates tasks individually.
    #     """
    #     if plan:
    #         self.current_plan = plan
    #     elif not self.current_plan:
    #         logger.error("No plan provided or loaded to process tasks in batch mode.")
    #         return False

    #     dev_tasks = [task for task in self.current_plan.tasks 
    #                  if task.agent_type == "dev_agent" and task.status == TaskStatus.PENDING]

    #     logger.info(f"Processing {len(dev_tasks)} development tasks in batch.")
    #     all_succeeded = True
    #     for task in dev_tasks:
    #         # execute_task now returns the updated Task object
    #         updated_task = await self.execute_task(task)
    #         if updated_task.status != TaskStatus.COMPLETED:
    #             all_succeeded = False
    #         
    #         # No need to call _save_updated_plan here as main.py (or the orchestrator)
    #         # should manage the overall plan persistence, having the full plan.
    #         # If this method is called, it assumes a batch mode and this agent might
    #         # be responsible for saving its progress to the plan file.
    #         # For the streaming model, main.py holds the single source of truth for the plan state.

    #     return all_succeeded

    # def _save_updated_plan(self):
    #     """
    #     Save the current plan with updated task statuses.
    #     This is relevant if this agent manages the full plan's persistence.
    #     In a streaming model, the orchestrator (main.py) might handle this.
    #     """
    #     if self.current_plan:
    #         plan_file = self.plan_dir / f"plan_{self.current_plan.id}.json"
    #         try:
    #             with open(plan_file, 'w', encoding='utf-8') as f:
    #                 json.dump(self.current_plan.to_dict(), f, indent=2, ensure_ascii=False)
    #             logger.info(f"DevAgent: Updated plan saved to {plan_file.name}")
    #         except Exception as e:
    #             logger.error(f"DevAgent: Failed to save updated plan {plan_file.name}: {e}", exc_info=True)

    def _get_latest_plan(self) -> Optional[Path]:
        """Find the most recent plan file in the parsed directory."""
        if not self.plan_dir.exists():
            return None
        plan_files = list(self.plan_dir.glob("plan_*.json"))
        if not plan_files:
            return None
        return max(plan_files, key=lambda p: p.stat().st_mtime)

    def load_current_plan(self) -> Optional[Plan]:
        """Load the most recent plan from the parsed plans directory."""
        plan_file = self._get_latest_plan()
        if not plan_file:
            logger.warning("No plan files found")
            return None
        
        try:
            with open(plan_file, 'r') as f:
                plan_data = json.load(f)
            self.current_plan = Plan(**plan_data)
            return self.current_plan
        except Exception as e:
            logger.error(f"Failed to load plan: {e}")
            return None

    async def process_plan_tasks(self, plan: Optional[Plan] = None):
        """Process all dev_agent tasks from the given plan or the current plan."""
        if plan:
            self.current_plan = plan
        elif not self.current_plan:
            logger.error("No plan provided or loaded to process tasks")
            return False

        dev_tasks = [task for task in self.current_plan.tasks 
                     if task.agent_type == "dev_agent" and task.status == TaskStatus.PENDING]

        logger.info(f"Processing {len(dev_tasks)} development tasks")
        for task in dev_tasks:
            success = await self.execute_task(task)
            if success:
                task.status = TaskStatus.COMPLETED
                self._save_updated_plan()
            else:
                task.status = TaskStatus.FAILED
                self._save_updated_plan()
                return False

        return True

    def _save_updated_plan(self):
        """Save the current plan with updated task statuses."""
        if self.current_plan:
            plan_file = self.plan_dir / f"plan_{self.current_plan.id}.json"
            with open(plan_file, 'w') as f:
                json.dump(self.current_plan.to_dict(), f, indent=2)

dev_agent = DevAgent()